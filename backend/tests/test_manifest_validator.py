"""Unit tests for VoxGuard ManifestValidator service.

Tests valid manifest parsing, missing required columns, invalid label/split enums,
path traversal sanitization, missing fields, and duplicate row detection.
"""

import os
import tempfile
import unittest

from app.evaluation.manifest_validator import ManifestValidator


class TestManifestValidator(unittest.TestCase):
    """Test suite for ManifestValidator CSV parsing and security safeguards."""

    def setUp(self):
        self.validator = ManifestValidator()
        self.temp_dir = tempfile.mkdtemp(prefix="voxguard_test_manifest_")

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_temp_csv(self, content: str) -> str:
        path = os.path.join(self.temp_dir, "test_manifest.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_1_valid_manifest(self):
        """1. Test valid manifest CSV parsing."""
        csv_text = (
            "file_path,label,speaker_id,source,source_id,split\n"
            "raw/human/1.wav,human,spk_1,src_1,src_id_1,train\n"
            "raw/synthetic/2.wav,synthetic,spk_2,src_2,src_id_2,validation\n"
        )
        path = self._create_temp_csv(csv_text)
        res = self.validator.validate_csv_manifest(path)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.total_samples, 2)
        self.assertEqual(len(res.errors), 0)

    def test_2_missing_column(self):
        """2. Test detection of missing required column."""
        csv_text = (
            "file_path,label,speaker_id,source,split\n"  # missing source_id
            "raw/human/1.wav,human,spk_1,src_1,train\n"
        )
        path = self._create_temp_csv(csv_text)
        res = self.validator.validate_csv_manifest(path)
        self.assertFalse(res.is_valid)
        self.assertIn("Missing required column in CSV header: 'source_id'", res.errors[0])

    def test_3_invalid_label(self):
        """3. Test detection of invalid label enum value."""
        csv_text = (
            "file_path,label,speaker_id,source,source_id,split\n"
            "raw/human/1.wav,fake_label,spk_1,src_1,src_id_1,train\n"
        )
        path = self._create_temp_csv(csv_text)
        res = self.validator.validate_csv_manifest(path)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("Invalid label" in e for e in res.errors))

    def test_4_missing_speaker_id(self):
        """4. Test detection of missing speaker ID value."""
        csv_text = (
            "file_path,label,speaker_id,source,source_id,split\n"
            "raw/human/1.wav,human,,src_1,src_id_1,train\n"  # empty speaker_id
        )
        path = self._create_temp_csv(csv_text)
        res = self.validator.validate_csv_manifest(path)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("Missing required values" in e for e in res.errors))

    def test_5_duplicate_file(self):
        """5. Test detection of duplicate file_path entries."""
        csv_text = (
            "file_path,label,speaker_id,source,source_id,split\n"
            "raw/human/1.wav,human,spk_1,src_1,src_id_1,train\n"
            "raw/human/1.wav,human,spk_1,src_1,src_id_1,validation\n"
        )
        path = self._create_temp_csv(csv_text)
        res = self.validator.validate_csv_manifest(path)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("Duplicate 'file_path'" in e for e in res.errors))

    def test_6_path_traversal_prevention(self):
        """15. Test detection and rejection of unsafe path traversal or absolute file paths."""
        csv_text = (
            "file_path,label,speaker_id,source,source_id,split\n"
            "../../etc/passwd,human,spk_1,src_1,src_id_1,train\n"
        )
        path = self._create_temp_csv(csv_text)
        res = self.validator.validate_csv_manifest(path)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("Unsafe or absolute file path" in e for e in res.errors))


if __name__ == "__main__":
    unittest.main()
