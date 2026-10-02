"""End-to-End Integration tests for VoxGuard Feature Dataset Generation (app.evaluation.feature_builder)."""

import csv
import json
import os
import tempfile
import unittest
import wave
import numpy as np

from app.evaluation.feature_builder import FeatureDatasetBuilder
from app.evaluation.dataset_integrity import DatasetIntegrity


def create_test_wav(filepath: str, duration_sec: float = 0.5, sample_rate: int = 16000, frequency: float = 440.0):
    """Generates a valid PCM 16-bit WAV file on disk for testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


class TestFeatureGeneration(unittest.TestCase):
    """Integration test suite for feature dataset generation pipeline."""

    def setUp(self):
        self.builder = FeatureDatasetBuilder()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audio_dir = os.path.join(self.temp_dir.name, "wavs")
        self.output_dir = os.path.join(self.temp_dir.name, "output_features")
        os.makedirs(self.audio_dir, exist_ok=True)

        # Create 3 tiny test WAV files
        self.wav1_path = os.path.join(self.audio_dir, "sample1.wav")
        self.wav2_path = os.path.join(self.audio_dir, "sample2.wav")
        self.wav3_path = os.path.join(self.audio_dir, "sample3.wav")

        create_test_wav(self.wav1_path, duration_sec=0.4, frequency=440.0)
        create_test_wav(self.wav2_path, duration_sec=0.6, frequency=880.0)
        create_test_wav(self.wav3_path, duration_sec=0.5, frequency=220.0)

        # Create valid manifest CSV
        self.manifest_path = os.path.join(self.temp_dir.name, "manifest.csv")
        self.manifest_rows = [
            {"file_path": "wavs/sample1.wav", "label": "human", "speaker_id": "spk_1", "source": "src1", "source_id": "v1", "split": "train"},
            {"file_path": "wavs/sample2.wav", "label": "synthetic", "speaker_id": "spk_2", "source": "src2", "source_id": "v2", "split": "validation"},
            {"file_path": "wavs/sample3.wav", "label": "human", "speaker_id": "spk_3", "source": "src1", "source_id": "v1", "split": "test"},
        ]
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_path", "label", "speaker_id", "source", "source_id", "split"])
            writer.writeheader()
            writer.writerows(self.manifest_rows)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_end_to_end_feature_generation(self):
        """Verify end-to-end feature generation produces valid .npz artifacts and metadata files."""
        res = self.builder.build_features(
            manifest_path=self.manifest_path,
            output_dir=self.output_dir,
            base_dir=self.temp_dir.name
        )

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["total_samples"], 3)
        self.assertEqual(res["successful_samples"], 3)
        self.assertEqual(res["failed_samples"], 0)

        # Verify generated files existence
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "features_manifest.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "features_manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "processing_report.json")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "statistics.json")))

        # Inspect a generated .npz feature artifact
        train_dir = os.path.join(self.output_dir, "train")
        npz_files = os.listdir(train_dir)
        self.assertEqual(len(npz_files), 1)

        npz_path = os.path.join(train_dir, npz_files[0])
        data = np.load(npz_path)

        self.assertIn("log_mel", data)
        self.assertIn("mfcc", data)
        self.assertIn("spectral_centroid", data)
        self.assertIn("spectral_rolloff", data)
        self.assertIn("label_code", data)

        self.assertEqual(data["log_mel"].shape[0], 80)
        self.assertEqual(data["mfcc"].shape[0], 20)
        self.assertEqual(int(data["label_code"]), 0)  # "human" -> 0

    def test_speaker_leakage_rejection(self):
        """Verify manifest with speaker overlap across splits is rejected."""
        leak_manifest_path = os.path.join(self.temp_dir.name, "leak_manifest.csv")
        leak_rows = [
            {"file_path": "wavs/sample1.wav", "label": "human", "speaker_id": "spk_SHARED", "source": "s", "source_id": "v", "split": "train"},
            {"file_path": "wavs/sample2.wav", "label": "synthetic", "speaker_id": "spk_SHARED", "source": "s", "source_id": "v", "split": "validation"},
        ]
        with open(leak_manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_path", "label", "speaker_id", "source", "source_id", "split"])
            writer.writeheader()
            writer.writerows(leak_rows)

        with self.assertRaises(ValueError) as ctx:
            self.builder.build_features(
                manifest_path=leak_manifest_path,
                output_dir=os.path.join(self.temp_dir.name, "leak_out"),
                base_dir=self.temp_dir.name
            )
        self.assertIn("Speaker leakage detected", str(ctx.exception))

    def test_dry_run_behavior(self):
        """Verify dry-run mode validates input without writing artifacts to disk."""
        dry_output = os.path.join(self.temp_dir.name, "dry_out")
        res = self.builder.build_features(
            manifest_path=self.manifest_path,
            output_dir=dry_output,
            base_dir=self.temp_dir.name,
            dry_run=True
        )

        self.assertEqual(res["status"], "DRY_RUN_SUCCESS")
        self.assertFalse(os.path.exists(dry_output))

    def test_deterministic_output(self):
        """Verify feature extraction on identical audio produces identical arrays and SHA-256 hashes."""
        out1 = os.path.join(self.temp_dir.name, "out1")
        out2 = os.path.join(self.temp_dir.name, "out2")

        self.builder.build_features(manifest_path=self.manifest_path, output_dir=out1, base_dir=self.temp_dir.name)
        self.builder.build_features(manifest_path=self.manifest_path, output_dir=out2, base_dir=self.temp_dir.name)

        with open(os.path.join(out1, "features_manifest.json")) as f1:
            json1 = json.load(f1)
        with open(os.path.join(out2, "features_manifest.json")) as f2:
            json2 = json.load(f2)

        self.assertEqual(json1["artifact_hashes"], json2["artifact_hashes"])

    def test_processing_failure_report(self):
        """Verify corrupt audio sample failure is recorded in processing_report.json without aborting valid samples."""
        bad_wav_path = os.path.join(self.audio_dir, "bad.wav")
        with open(bad_wav_path, "wb") as f:
            f.write(b"CORRUPTED_HEADER_BYTES")

        fail_manifest = os.path.join(self.temp_dir.name, "fail_manifest.csv")
        rows = [
            {"file_path": "wavs/sample1.wav", "label": "human", "speaker_id": "spk_1", "source": "s", "source_id": "v", "split": "train"},
            {"file_path": "wavs/bad.wav", "label": "synthetic", "speaker_id": "spk_2", "source": "s", "source_id": "v", "split": "test"},
        ]
        with open(fail_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_path", "label", "speaker_id", "source", "source_id", "split"])
            writer.writeheader()
            writer.writerows(rows)

        out_fail = os.path.join(self.temp_dir.name, "fail_out")
        res = self.builder.build_features(manifest_path=fail_manifest, output_dir=out_fail, base_dir=self.temp_dir.name)

        self.assertEqual(res["total_samples"], 2)
        self.assertEqual(res["successful_samples"], 1)
        self.assertEqual(res["failed_samples"], 1)

        with open(os.path.join(out_fail, "processing_report.json")) as f:
            rep = json.load(f)

        self.assertEqual(rep["failed_samples"], 1)
        self.assertEqual(len(rep["failures"]), 1)
        self.assertIn("error_code", rep["failures"][0])


if __name__ == "__main__":
    unittest.main()
