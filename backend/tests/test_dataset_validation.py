"""Unit tests for VoxGuard Dataset Stream Validation and Audio Auditing.

Tests audio decodability, duration statistics computation, corrupted file recovery,
and unsupported media format handling using synthetic WAV audio fixtures.
"""

import os
import shutil
import tempfile
import unittest
import wave

from app.evaluation.dataset_report import DatasetReportBuilder
from app.evaluation.manifest_validator import ManifestValidator


class TestDatasetValidation(unittest.TestCase):
    """Test suite for dataset audio auditing and duration statistics."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="voxguard_test_audio_ds_")
        self.builder = DatasetReportBuilder()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_sample_wav(self, filename: str, duration_sec: float = 1.0) -> str:
        path = os.path.join(self.temp_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sample_rate = 16000
        n_samples = int(sample_rate * duration_sec)
        
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(b"\x00\x00" * n_samples)
        return path

    def test_audio_auditing_and_duration_stats(self):
        """12. Verify valid audio stream auditing and duration stats (min, max, mean, median)."""
        self._create_sample_wav("human/1.wav", duration_sec=1.0)
        self._create_sample_wav("human/2.wav", duration_sec=3.0)
        self._create_sample_wav("synthetic/3.wav", duration_sec=2.0)

        manifest_text = (
            "file_path,label,speaker_id,source,source_id,split\n"
            "human/1.wav,human,spk_1,src_1,src_id_1,train\n"
            "human/2.wav,human,spk_2,src_1,src_id_2,validation\n"
            "synthetic/3.wav,synthetic,spk_3,src_2,src_id_3,test\n"
        )
        manifest_path = os.path.join(self.temp_dir, "manifest.csv")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest_text)

        report = self.builder.build_report(manifest_path, base_dir=self.temp_dir, audit_audio=True)
        self.assertEqual(report.manifest_status, "VALID")
        self.assertEqual(report.audio_audit_status, "VALID")
        self.assertEqual(report.speaker_disjoint_status, "PASS")
        self.assertEqual(report.overall_status, "READY")
        self.assertEqual(report.valid_audio_count, 3)

        ds = report.duration_stats
        self.assertEqual(ds["min"], 1.0)
        self.assertEqual(ds["max"], 3.0)
        self.assertEqual(ds["mean"], 2.0)
        self.assertEqual(ds["median"], 2.0)

    def test_corrupted_and_missing_audio_handling(self):
        """6 & 7. Verify detection of missing or corrupted audio streams."""
        # Create a corrupted non-WAV file
        bad_path = os.path.join(self.temp_dir, "bad.wav")
        with open(bad_path, "wb") as f:
            f.write(b"CORRUPTED_NOT_AUDIO_HEADER")

        manifest_text = (
            "file_path,label,speaker_id,source,source_id,split\n"
            "bad.wav,human,spk_1,src_1,src_id_1,train\n"
            "missing.wav,synthetic,spk_2,src_2,src_id_2,test\n"
        )
        manifest_path = os.path.join(self.temp_dir, "manifest.csv")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest_text)

        report = self.builder.build_report(manifest_path, base_dir=self.temp_dir, audit_audio=True)
        self.assertEqual(report.manifest_status, "VALID")
        self.assertEqual(report.audio_audit_status, "INVALID")
        self.assertEqual(report.overall_status, "NOT_READY")
        self.assertEqual(report.invalid_audio_count, 2)


if __name__ == "__main__":
    unittest.main()
