"""Unit tests for VoxGuard Audio Feature Extractor Service.

Tests Log-Mel Spectrogram, MFCC, Spectral Centroid, and Spectral Rolloff extraction,
verifying output shapes, dtypes, non-finite absence, and error handling.
Inherits from unittest.TestCase for standalone and pytest compatibility.
"""

import math
import unittest

import numpy as np

from app.audio.preprocessor import PreprocessedAudio
from app.features.extractor import (
    AudioFeatureExtractor,
    AudioFeatures,
    FeatureConfig,
    FeatureExtractionError,
    FeatureResult,
)


class TestAudioFeatureExtractor(unittest.TestCase):
    """Test suite for AudioFeatureExtractor service."""

    def setUp(self):
        self.config = FeatureConfig(
            n_fft=1024,
            hop_length=512,
            n_mels=80,
            n_mfcc=20,
            f_min=0.0,
            f_max=8000.0,
            roll_percent=0.85
        )
        self.extractor = AudioFeatureExtractor(config=self.config)

    def _generate_synthetic_preprocessed(
        self, duration_sec: float = 1.0, freq_hz: float = 440.0
    ) -> PreprocessedAudio:
        """Generates synthetic 16kHz mono float32 sine wave PreprocessedAudio object."""
        n_samples = int(duration_sec * 16000)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        pcm = (0.7 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)

        return PreprocessedAudio(
            pcm_data=pcm,
            sample_rate=16000,
            channels=1,
            num_frames=n_samples,
            duration_seconds=duration_sec,
            metadata={"synthetic": True}
        )

    def test_extract_all_features(self):
        preprocessed = self._generate_synthetic_preprocessed(duration_sec=1.0)
        features: AudioFeatures = self.extractor.extract_all(preprocessed)

        self.assertIsInstance(features, AudioFeatures)
        self.assertEqual(features.sample_rate, 16000)
        self.assertEqual(features.duration_seconds, 1.0)

        # Check Log-Mel Spectrogram
        log_mel = features.log_mel_spectrogram
        self.assertEqual(log_mel.feature_name, "log_mel_spectrogram")
        self.assertEqual(log_mel.shape[0], 80)  # n_mels = 80
        self.assertEqual(log_mel.dtype, "float32")
        self.assertTrue(np.isfinite(log_mel.data).all())

        # Check MFCC
        mfcc = features.mfcc
        self.assertEqual(mfcc.feature_name, "mfcc")
        self.assertEqual(mfcc.shape[0], 20)  # n_mfcc = 20
        self.assertEqual(mfcc.dtype, "float32")
        self.assertTrue(np.isfinite(mfcc.data).all())

        # Check Spectral Centroid
        centroid = features.spectral_centroid
        self.assertEqual(centroid.feature_name, "spectral_centroid")
        self.assertEqual(centroid.shape[0], 1)
        self.assertEqual(centroid.dtype, "float32")
        self.assertTrue(np.isfinite(centroid.data).all())

        # Check Spectral Rolloff
        rolloff = features.spectral_rolloff
        self.assertEqual(rolloff.feature_name, "spectral_rolloff")
        self.assertEqual(rolloff.shape[0], 1)
        self.assertEqual(rolloff.dtype, "float32")
        self.assertTrue(np.isfinite(rolloff.data).all())

    def test_spectral_centroid_value_range(self):
        # 1000 Hz sine wave should yield spectral centroid near 1000 Hz
        preprocessed = self._generate_synthetic_preprocessed(duration_sec=1.0, freq_hz=1000.0)
        features = self.extractor.extract_all(preprocessed)

        centroid_vals = features.spectral_centroid.data[0]
        mean_centroid = float(np.mean(centroid_vals))
        
        # Centroid should be within 1000 Hz neighborhood (tolerance for FFT bin resolution)
        self.assertTrue(math.isclose(mean_centroid, 1000.0, abs_tol=200.0))

    def test_short_audio_padding_handling(self):
        # Audio shorter than n_fft (e.g. 500 samples < 1024)
        short_pcm = np.ones(500, dtype=np.float32) * 0.2
        preprocessed = PreprocessedAudio(
            pcm_data=short_pcm,
            sample_rate=16000,
            channels=1,
            num_frames=500,
            duration_seconds=500 / 16000.0
        )

        features = self.extractor.extract_all(preprocessed)
        self.assertGreater(features.log_mel_spectrogram.shape[1], 0)

    def test_empty_audio_raises_error(self):
        empty_preprocessed = PreprocessedAudio(
            pcm_data=np.array([], dtype=np.float32),
            sample_rate=16000,
            channels=1,
            num_frames=0,
            duration_seconds=0.0
        )

        with self.assertRaises(FeatureExtractionError):
            self.extractor.extract_all(empty_preprocessed)


if __name__ == "__main__":
    unittest.main()
