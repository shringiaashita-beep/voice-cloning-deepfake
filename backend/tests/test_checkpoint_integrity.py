"""Unit Tests for VoxGuard Checkpoint Integrity and SHA-256 Validation.

Validates checkpoint hash calculation, ModelManifest serialization, and
checksum integrity enforcement via ModelValidator.
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest

from app.models.model_manifest import ModelManifest
from app.models.model_validator import ModelValidator


class TestCheckpointIntegrity(unittest.TestCase):
    """Test suite for checkpoint integrity and SHA-256 manifest validation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.test_dir, "test_model.pt")
        self.manifest_path = os.path.join(self.test_dir, "manifest.json")

        # Create dummy checkpoint binary data
        self.dummy_bytes = b"VoxGuard_Synthetic_PyTorch_Model_Weights_V1_Checksum_Data"
        with open(self.checkpoint_path, "wb") as f:
            f.write(self.dummy_bytes)

        # Compute valid SHA-256 digest
        self.expected_sha256 = hashlib.sha256(self.dummy_bytes).hexdigest()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_sha256_checksum_verification(self):
        """Verify ModelValidator validates valid SHA-256 checksum."""
        validator = ModelValidator()
        res = validator.validate_checkpoint_file(self.checkpoint_path, expected_checksum=self.expected_sha256)

        self.assertTrue(res.is_valid)
        self.assertIsNone(res.error_message)

    def test_corrupt_checksum_rejection(self):
        """Verify ModelValidator rejects checkpoint with invalid SHA-256 hash."""
        validator = ModelValidator()
        invalid_checksum = "0" * 64
        res = validator.validate_checkpoint_file(self.checkpoint_path, expected_checksum=invalid_checksum)

        self.assertFalse(res.is_valid)
        self.assertIn("checksum mismatch", res.error_message.lower())

    def test_manifest_roundtrip_and_fields(self):
        """Verify ModelManifest serialization and deserialization."""
        manifest = ModelManifest(
            model_name="VoxGuardAcousticNet",
            model_version="1.0.0",
            model_type="pytorch",
            architecture="VoxGuardAcousticNet",
            checkpoint_path=self.checkpoint_path,
            checksum=self.expected_sha256
        )

        manifest_dict = manifest.to_dict()
        manifest_restored = ModelManifest.from_dict(manifest_dict)

        self.assertEqual(manifest_restored.model_name, "VoxGuardAcousticNet")
        self.assertEqual(manifest_restored.checksum, self.expected_sha256)
        self.assertEqual(manifest_restored.architecture, "VoxGuardAcousticNet")


if __name__ == "__main__":
    unittest.main()
