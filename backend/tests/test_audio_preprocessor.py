"""Unit tests for VoxGuard Audio Preprocessor Module.

Validates resampling (8kHz, 16kHz, 44.1kHz -> 16kHz), mono conversion,
peak normalization, deterministic execution, and numerical sanity.
Inherits from unittest.TestCase for standalone and pytest compatibility.
"""

import math
import unittest

import numpy as np

from app.audio.decoder import DecodedAudio
from app.audio.preprocessor import AudioPreprocessor, AudioPreprocessingError, PreprocessedAudio


class TestAudioPreprocessor(unittest.TestCase):
    """Test suite for AudioPreprocessor service."""

    def setUp(self):
        self.preprocessor = AudioPreprocessor()

    def test_resample_8khz_to_16khz(self):
        # 1 second of 8kHz mono sine wave
        t = np.linspace(0, 1.0, 8000, endpoint=False)
        pcm_8k = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        decoded = DecodedAudio(
            pcm_data=pcm_8k,
            sample_rate=8000,
            channels=1,
            num_frames=8000,
            duration_seconds=1.0,
            format="wav"
        )

        result: PreprocessedAudio = self.preprocessor.process(decoded)

        # Numerical Assertions
        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(result.channels, 1)
        self.assertEqual(result.pcm_data.dtype, np.float32)
        self.assertTrue(np.isfinite(result.pcm_data).all())
        self.assertFalse(np.isnan(result.pcm_data).any())
        self.assertFalse(np.isinf(result.pcm_data).any())
        self.assertGreater(result.num_frames, 0)
        self.assertTrue(math.isclose(result.duration_seconds, 1.0, abs_tol=0.05))

    def test_resample_16khz_to_16khz_noop(self):
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        pcm_16k = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        decoded = DecodedAudio(
            pcm_data=pcm_16k,
            sample_rate=16000,
            channels=1,
            num_frames=16000,
            duration_seconds=1.0,
            format="wav"
        )

        result = self.preprocessor.process(decoded)

        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(result.num_frames, 16000)
        self.assertEqual(result.pcm_data.dtype, np.float32)
        self.assertTrue(np.isfinite(result.pcm_data).all())

    def test_resample_44100hz_to_16khz(self):
        t = np.linspace(0, 2.0, 88200, endpoint=False)
        pcm_44k = (0.9 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        decoded = DecodedAudio(
            pcm_data=pcm_44k,
            sample_rate=44100,
            channels=1,
            num_frames=88200,
            duration_seconds=2.0,
            format="wav"
        )

        result = self.preprocessor.process(decoded)

        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(result.channels, 1)
        self.assertEqual(result.pcm_data.dtype, np.float32)
        self.assertTrue(np.isfinite(result.pcm_data).all())
        # Expected frames: ~32000 frames for 2 seconds at 16kHz
        self.assertTrue(math.isclose(result.num_frames, 32000, abs_tol=10))
        self.assertTrue(math.isclose(result.duration_seconds, 2.0, abs_tol=0.05))

    def test_stereo_to_mono_conversion(self):
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        ch1 = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        ch2 = (0.3 * np.cos(2 * np.pi * 880 * t)).astype(np.float32)
        stereo_pcm = np.column_stack((ch1, ch2))

        decoded = DecodedAudio(
            pcm_data=stereo_pcm,
            sample_rate=16000,
            channels=2,
            num_frames=16000,
            duration_seconds=1.0,
            format="wav"
        )

        result = self.preprocessor.process(decoded)

        self.assertEqual(result.channels, 1)
        self.assertEqual(result.pcm_data.ndim, 1)
        self.assertEqual(result.sample_rate, 16000)
        self.assertTrue(result.metadata["channel_converted"])

    def test_mono_remains_mono(self):
        pcm_mono = np.zeros(16000, dtype=np.float32)
        decoded = DecodedAudio(
            pcm_data=pcm_mono,
            sample_rate=16000,
            channels=1,
            num_frames=16000,
            duration_seconds=1.0,
            format="wav"
        )

        result = self.preprocessor.process(decoded)

        self.assertEqual(result.channels, 1)
        self.assertEqual(result.pcm_data.ndim, 1)
        self.assertFalse(result.metadata["channel_converted"])

    def test_empty_audio_handling(self):
        decoded = DecodedAudio(
            pcm_data=np.array([], dtype=np.float32),
            sample_rate=16000,
            channels=1,
            num_frames=0,
            duration_seconds=0.0,
            format="wav"
        )

        with self.assertRaises(AudioPreprocessingError):
            self.preprocessor.process(decoded)

    def test_very_short_audio_handling(self):
        # 16 samples (0.001 seconds at 16kHz)
        short_pcm = (0.4 * np.ones(16)).astype(np.float32)
        decoded = DecodedAudio(
            pcm_data=short_pcm,
            sample_rate=16000,
            channels=1,
            num_frames=16,
            duration_seconds=0.001,
            format="wav"
        )

        result = self.preprocessor.process(decoded)

        self.assertGreater(result.num_frames, 0)
        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(result.pcm_data.dtype, np.float32)

    def test_peak_normalization_scaling(self):
        # Peak amplitude 2.5 should scale max amplitude down to exactly 1.0
        pcm_raw = np.array([-2.5, 0.0, 1.25, 2.5], dtype=np.float32)
        decoded = DecodedAudio(
            pcm_data=pcm_raw,
            sample_rate=16000,
            channels=1,
            num_frames=4,
            duration_seconds=0.00025,
            format="wav"
        )

        result = self.preprocessor.process(decoded)

        max_val = np.max(np.abs(result.pcm_data))
        self.assertTrue(math.isclose(max_val, 1.0, abs_tol=1e-5))

    def test_deterministic_output(self):
        np.random.seed(42)
        random_pcm = np.random.uniform(-0.8, 0.8, 22050).astype(np.float32)
        decoded = DecodedAudio(
            pcm_data=random_pcm,
            sample_rate=22050,
            channels=1,
            num_frames=22050,
            duration_seconds=1.0,
            format="wav"
        )

        run1 = self.preprocessor.process(decoded)
        run2 = self.preprocessor.process(decoded)

        self.assertTrue(np.array_equal(run1.pcm_data, run2.pcm_data))
        self.assertEqual(run1.metadata, run2.metadata)


if __name__ == "__main__":
    unittest.main()
