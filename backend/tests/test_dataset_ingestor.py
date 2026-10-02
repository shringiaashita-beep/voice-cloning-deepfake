"""Unit tests for VoxGuard Dataset Ingestor (app.evaluation.dataset_ingestor)."""

import os
import tempfile
import unittest
import wave
import numpy as np

from app.evaluation.dataset_ingestor import DatasetIngestor, DatasetIngestionError, DatasetSample
from app.evaluation.manifest_validator import ManifestRow
from app.features.extractor import AudioFeatures


def create_test_wav(filepath: str, duration_sec: float = 0.5, sample_rate: int = 16000):
    """Generates a valid PCM 16-bit WAV file on disk for testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


class TestDatasetIngestor(unittest.TestCase):
    """Test suite for DatasetIngestor audio processing stream."""

    def setUp(self):
        self.ingestor = DatasetIngestor()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.wav_path = os.path.join(self.temp_dir.name, "test_audio.wav")
        create_test_wav(self.wav_path, duration_sec=0.5)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_sample_ingestion(self):
        """Verify successful ingestion of a valid audio sample."""
        row = ManifestRow(
            file_path="test_audio.wav",
            label="human",
            speaker_id="spk_001",
            source="LibriSpeech",
            source_id="v_001",
            split="train"
        )
        sample, features = self.ingestor.process_sample_row(row, base_dir=self.temp_dir.name)

        self.assertIsInstance(sample, DatasetSample)
        self.assertIsInstance(features, AudioFeatures)
        self.assertEqual(sample.label, "human")
        self.assertEqual(sample.speaker_id, "spk_001")
        self.assertIn("log_mel_shape", sample.features)
        self.assertEqual(features.log_mel_spectrogram.data.shape[0], 80)
        self.assertEqual(features.mfcc.data.shape[0], 20)

    def test_invalid_label_rejection(self):
        """Verify invalid dataset label raises DatasetIngestionError."""
        row = ManifestRow(
            file_path="test_audio.wav",
            label="unknown_label",
            speaker_id="spk_001",
            source="Test",
            source_id="t_1",
            split="train"
        )
        with self.assertRaises(DatasetIngestionError) as ctx:
            self.ingestor.process_sample_row(row, base_dir=self.temp_dir.name)
        self.assertEqual(ctx.exception.code, "INVALID_LABEL")

    def test_missing_audio_rejection(self):
        """Verify non-existent audio file path raises DatasetIngestionError."""
        row = ManifestRow(
            file_path="missing_file.wav",
            label="human",
            speaker_id="spk_001",
            source="Test",
            source_id="t_1",
            split="train"
        )
        with self.assertRaises(DatasetIngestionError) as ctx:
            self.ingestor.process_sample_row(row, base_dir=self.temp_dir.name)
        self.assertEqual(ctx.exception.code, "FILE_NOT_FOUND")

    def test_corrupted_audio_rejection(self):
        """Verify corrupted audio file bytes raise DatasetIngestionError."""
        bad_wav_path = os.path.join(self.temp_dir.name, "corrupt.wav")
        with open(bad_wav_path, "wb") as f:
            f.write(b"NOT_A_WAV_FILE_HEADER_CORRUPTED_BYTES")

        row = ManifestRow(
            file_path="corrupt.wav",
            label="human",
            speaker_id="spk_001",
            source="Test",
            source_id="t_1",
            split="train"
        )
        with self.assertRaises(DatasetIngestionError) as ctx:
            self.ingestor.process_sample_row(row, base_dir=self.temp_dir.name)
        self.assertIn(ctx.exception.code, ["DECODING_FAILED", "DECODING_ERROR"])

    def test_path_traversal_rejection(self):
        """Verify path traversal attempts raise DatasetIngestionError."""
        row = ManifestRow(
            file_path="../secret.wav",
            label="human",
            speaker_id="spk_001",
            source="Test",
            source_id="t_1",
            split="train"
        )
        with self.assertRaises(DatasetIngestionError) as ctx:
            self.ingestor.process_sample_row(row, base_dir=self.temp_dir.name)
        self.assertEqual(ctx.exception.code, "PATH_TRAVERSAL")

    def test_variable_length_audio(self):
        """Verify audio files with different durations produce different feature frame counts without error."""
        wav_short = os.path.join(self.temp_dir.name, "short.wav")
        wav_long = os.path.join(self.temp_dir.name, "long.wav")

        create_test_wav(wav_short, duration_sec=0.3)
        create_test_wav(wav_long, duration_sec=1.0)

        row_short = ManifestRow("short.wav", "human", "spk_1", "src", "s1", "train")
        row_long = ManifestRow("long.wav", "synthetic", "spk_2", "src", "s2", "train")

        sample_short, feat_short = self.ingestor.process_sample_row(row_short, base_dir=self.temp_dir.name)
        sample_long, feat_long = self.ingestor.process_sample_row(row_long, base_dir=self.temp_dir.name)

        t_short = feat_short.log_mel_spectrogram.data.shape[1]
        t_long = feat_long.log_mel_spectrogram.data.shape[1]

        self.assertGreater(t_long, t_short)
        self.assertNotEqual(t_short, t_long)


if __name__ == "__main__":
    unittest.main()
