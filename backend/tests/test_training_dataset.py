"""Unit tests for VoxGuard Training Feature Dataset Loader (app.training.dataset)."""

import csv
import os
import tempfile
import unittest
import numpy as np

from app.training.config import TrainingConfig
from app.training.dataset import FeatureDataset, DatasetLoaderError


def create_dummy_npz(filepath: str, t_frames: int = 300, label_code: int = 0, is_corrupt: bool = False, bad_array: str = None):
    """Creates a dummy .npz feature artifact file on disk."""
    if is_corrupt:
        with open(filepath, "wb") as f:
            f.write(b"NOT_A_VALID_NPZ_FILE_BYTES")
        return

    log_mel = np.ones((80, t_frames), dtype=np.float32)
    mfcc = np.ones((20, t_frames), dtype=np.float32)
    centroid = np.ones((1, t_frames), dtype=np.float32)
    rolloff = np.ones((1, t_frames), dtype=np.float32)

    if bad_array == "wrong_mel_shape":
        log_mel = np.ones((64, t_frames), dtype=np.float32)
    elif bad_array == "wrong_mfcc_shape":
        mfcc = np.ones((13, t_frames), dtype=np.float32)
    elif bad_array == "inconsistent_t":
        mfcc = np.ones((20, t_frames + 50), dtype=np.float32)
    elif bad_array == "nan":
        log_mel[0, 0] = np.nan
    elif bad_array == "inf":
        mfcc[0, 0] = np.inf
    elif bad_array == "wrong_dtype":
        log_mel = log_mel.astype(np.float64)

    np.savez(
        filepath,
        log_mel=log_mel,
        mfcc=mfcc,
        spectral_centroid=centroid,
        spectral_rolloff=rolloff,
        label_code=np.int64(label_code),
        speaker_id=np.array("spk_001")
    )


class TestTrainingDataset(unittest.TestCase):
    """Test suite for FeatureDataset loader."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.feature_dir = os.path.join(self.temp_dir.name, "features")
        os.makedirs(os.path.join(self.feature_dir, "train"), exist_ok=True)

        self.npz_path1 = os.path.join(self.feature_dir, "train", "sample1.npz")
        self.npz_path2 = os.path.join(self.feature_dir, "train", "sample2.npz")

        create_dummy_npz(self.npz_path1, t_frames=300, label_code=0)
        create_dummy_npz(self.npz_path2, t_frames=150, label_code=1)

        self.manifest_path = os.path.join(self.temp_dir.name, "features_manifest.csv")
        self.rows = [
            {"feature_path": "train/sample1.npz", "label": "human", "speaker_id": "spk_1", "source": "src1", "source_id": "v1", "split": "train"},
            {"feature_path": "train/sample2.npz", "label": "synthetic", "speaker_id": "spk_2", "source": "src2", "source_id": "v2", "split": "train"},
        ]
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["feature_path", "label", "speaker_id", "source", "source_id", "split"])
            writer.writeheader()
            writer.writerows(self.rows)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_npz_loading_and_label_encoding(self):
        """Verify successful NPZ loading, windowing, and label encoding."""
        dataset = FeatureDataset(manifest_path=self.manifest_path, feature_dir=self.feature_dir, split="train")

        # sample1 (T=300 -> 2 windows), sample2 (T=150 -> 1 window) => Total 3 windows
        self.assertEqual(len(dataset), 3)

        sample_0 = dataset[0]
        self.assertEqual(sample_0.label, 0)
        self.assertEqual(sample_0.label_str, "human")
        self.assertEqual(sample_0.speaker_id, "spk_1")

        # Find synthetic sample
        synth_samples = [s for s in dataset if s.label_str == "synthetic"]
        self.assertEqual(len(synth_samples), 1)
        self.assertEqual(synth_samples[0].label, 1)

    def test_invalid_npz_loading(self):
        """Verify corrupt .npz file raises DatasetLoaderError."""
        bad_npz = os.path.join(self.feature_dir, "train", "corrupt.npz")
        create_dummy_npz(bad_npz, is_corrupt=True)

        bad_manifest = os.path.join(self.temp_dir.name, "bad_manifest.csv")
        with open(bad_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["feature_path", "label", "speaker_id", "source", "source_id", "split"])
            writer.writeheader()
            writer.writerow({"feature_path": "train/corrupt.npz", "label": "human", "speaker_id": "spk_1", "source": "src", "source_id": "v", "split": "train"})

        with self.assertRaises(DatasetLoaderError):
            FeatureDataset(manifest_path=bad_manifest, feature_dir=self.feature_dir, split="train")

    def test_wrong_log_mel_shape(self):
        """Verify wrong Log-Mel shape (64 instead of 80) raises DatasetLoaderError."""
        bad_npz = os.path.join(self.feature_dir, "train", "bad_mel.npz")
        create_dummy_npz(bad_npz, bad_array="wrong_mel_shape")

        bad_manifest = os.path.join(self.temp_dir.name, "bad_mel_manifest.csv")
        with open(bad_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["feature_path", "label", "speaker_id", "source", "source_id", "split"])
            writer.writeheader()
            writer.writerow({"feature_path": "train/bad_mel.npz", "label": "human", "speaker_id": "spk_1", "source": "src", "source_id": "v", "split": "train"})

        with self.assertRaises(DatasetLoaderError):
            FeatureDataset(manifest_path=bad_manifest, feature_dir=self.feature_dir, split="train")

    def test_inconsistent_t(self):
        """Verify inconsistent temporal frame length raises DatasetLoaderError."""
        bad_npz = os.path.join(self.feature_dir, "train", "inconsistent.npz")
        create_dummy_npz(bad_npz, bad_array="inconsistent_t")

        bad_manifest = os.path.join(self.temp_dir.name, "inconsistent_manifest.csv")
        with open(bad_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["feature_path", "label", "speaker_id", "source", "source_id", "split"])
            writer.writeheader()
            writer.writerow({"feature_path": "train/inconsistent.npz", "label": "human", "speaker_id": "spk_1", "source": "src", "source_id": "v", "split": "train"})

        with self.assertRaises(DatasetLoaderError):
            FeatureDataset(manifest_path=bad_manifest, feature_dir=self.feature_dir, split="train")

    def test_nan_detection(self):
        """Verify NaN array values raise DatasetLoaderError."""
        bad_npz = os.path.join(self.feature_dir, "train", "nan.npz")
        create_dummy_npz(bad_npz, bad_array="nan")

        bad_manifest = os.path.join(self.temp_dir.name, "nan_manifest.csv")
        with open(bad_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["feature_path", "label", "speaker_id", "source", "source_id", "split"])
            writer.writeheader()
            writer.writerow({"feature_path": "train/nan.npz", "label": "human", "speaker_id": "spk_1", "source": "src", "source_id": "v", "split": "train"})

        with self.assertRaises(DatasetLoaderError):
            FeatureDataset(manifest_path=bad_manifest, feature_dir=self.feature_dir, split="train")

    def test_path_traversal_rejection(self):
        """Verify relative path with '../' raises DatasetLoaderError."""
        bad_manifest = os.path.join(self.temp_dir.name, "trav_manifest.csv")
        with open(bad_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["feature_path", "label", "speaker_id", "source", "source_id", "split"])
            writer.writeheader()
            writer.writerow({"feature_path": "../secret.npz", "label": "human", "speaker_id": "spk_1", "source": "src", "source_id": "v", "split": "train"})

        with self.assertRaises(DatasetLoaderError):
            FeatureDataset(manifest_path=bad_manifest, feature_dir=self.feature_dir, split="train")

    def test_absolute_path_rejection(self):
        """Verify absolute path raises DatasetLoaderError."""
        bad_manifest = os.path.join(self.temp_dir.name, "abs_manifest.csv")
        with open(bad_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["feature_path", "label", "speaker_id", "source", "source_id", "split"])
            writer.writeheader()
            writer.writerow({"feature_path": "C:/Windows/System32/file.npz", "label": "human", "speaker_id": "spk_1", "source": "src", "source_id": "v", "split": "train"})

        with self.assertRaises(DatasetLoaderError):
            FeatureDataset(manifest_path=bad_manifest, feature_dir=self.feature_dir, split="train")


if __name__ == "__main__":
    unittest.main()
