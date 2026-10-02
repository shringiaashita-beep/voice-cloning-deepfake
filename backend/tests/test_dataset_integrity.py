"""Unit tests for VoxGuard Dataset Integrity (app.evaluation.dataset_integrity)."""

import hashlib
import json
import os
import tempfile
import unittest

from app.evaluation.dataset_integrity import DatasetIntegrity


class TestDatasetIntegrity(unittest.TestCase):
    """Test suite for DatasetIntegrity checksums and metadata JSON generation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dummy_path = os.path.join(self.temp_dir.name, "sample.npz")
        self.dummy_bytes = b"VOXGUARD_TEST_NPZ_BYTES_PAYLOAD"
        with open(self.dummy_path, "wb") as f:
            f.write(self.dummy_bytes)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sha256_generation(self):
        """Verify SHA-256 computation produces correct hex string."""
        computed_hash = DatasetIntegrity.compute_file_sha256(self.dummy_path)
        expected_hash = hashlib.sha256(self.dummy_bytes).hexdigest()

        self.assertEqual(computed_hash, expected_hash)
        self.assertEqual(len(computed_hash), 64)

    def test_build_features_manifest_json(self):
        """Verify features_manifest.json metadata structure and content."""
        hashes = {
            "train/sample_1.npz": "a" * 64,
            "validation/sample_2.npz": "b" * 64
        }
        manifest_data = DatasetIntegrity.build_features_manifest_json(
            artifact_hashes=hashes,
            dataset_version="1.0.0",
            feature_pipeline_version="voxguard-features-0.1.0"
        )

        self.assertEqual(manifest_data["dataset_version"], "1.0.0")
        self.assertEqual(manifest_data["feature_pipeline_version"], "voxguard-features-0.1.0")
        self.assertEqual(manifest_data["sample_count"], 2)
        self.assertIn("generation_timestamp", manifest_data)
        self.assertEqual(manifest_data["artifact_hashes"]["train/sample_1.npz"], "a" * 64)

    def test_no_absolute_paths_in_generated_metadata(self):
        """Verify absolute paths are sanitized and excluded from manifest metadata."""
        abs_key = r"C:\Users\Secret\VoxGuard\datasets\train\sample.npz"
        hashes = {abs_key: "c" * 64}

        manifest_data = DatasetIntegrity.build_features_manifest_json(hashes)

        for k in manifest_data["artifact_hashes"].keys():
            self.assertNotIn("C:", k)
            self.assertNotIn("Users", k)
            self.assertNotIn("\\", k)


if __name__ == "__main__":
    unittest.main()
