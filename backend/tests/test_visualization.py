"""Unit tests for VoxGuard Visualization Payload Builder.

Tests waveform peak envelope generation (~300 points), compact 40x120 spectrogram grid,
[0.0, 1.0] normalization, zero division safety, and JSON serialization compatibility.
Inherits from unittest.TestCase for standalone and pytest compatibility.
"""

import json
import unittest

import numpy as np

from app.audio.preprocessor import PreprocessedAudio
from app.features.extractor import AudioFeatureExtractor, FeatureConfig
from app.features.visualization import (
    SpectrogramPayload,
    VisualizationBuilder,
    VisualizationPayload,
    WaveformPayload,
)


class TestVisualizationBuilder(unittest.TestCase):
    """Test suite for VisualizationBuilder service."""

    def setUp(self):
        self.builder = VisualizationBuilder(
            waveform_points=300,
            spectrogram_freq_bins=40,
            spectrogram_time_bins=120
        )
        self.extractor = AudioFeatureExtractor()

    def test_build_waveform_peaks(self):
        # Generate 16000 samples (1 sec) sine wave with peak amplitude 0.8
        n_samples = 16000
        t = np.linspace(0, 1.0, n_samples, endpoint=False)
        pcm = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        # Insert a transient peak spike (+0.95) at sample 5000
        pcm[5000] = 0.95

        waveform: WaveformPayload = self.builder.build_waveform(pcm, duration_seconds=1.0)

        self.assertEqual(waveform.target_points, 300)
        self.assertEqual(len(waveform.min_peaks), 300)
        self.assertEqual(len(waveform.max_peaks), 300)
        self.assertEqual(len(waveform.peak_envelope), 300)
        self.assertEqual(waveform.duration_seconds, 1.0)

        # Verify peak envelope values are bounded in [0.0, 1.0]
        for val in waveform.peak_envelope:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

        # Transient peak spike should be captured in max peaks
        max_recorded_peak = max(waveform.max_peaks)
        self.assertAlmostEqual(max_recorded_peak, 0.95, delta=0.01)

    def test_build_spectrogram_grid(self):
        # Create dummy log-mel matrix of shape (80, 200)
        log_mel_data = np.random.uniform(-80.0, 0.0, (80, 200)).astype(np.float32)

        spectrogram: SpectrogramPayload = self.builder.build_spectrogram(log_mel_data)

        self.assertEqual(spectrogram.freq_bins, 40)
        self.assertEqual(spectrogram.time_bins, 120)
        self.assertEqual(len(spectrogram.data_grid), 40)
        self.assertEqual(len(spectrogram.data_grid[0]), 120)

        # Check normalization bounds strictly [0.0, 1.0]
        for row in spectrogram.data_grid:
            for val in row:
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)

    def test_json_serialization(self):
        # Create full preprocessed + feature objects
        pcm = (0.5 * np.sin(np.linspace(0, 10, 16000))).astype(np.float32)
        preprocessed = PreprocessedAudio(
            pcm_data=pcm, sample_rate=16000, channels=1, num_frames=16000, duration_seconds=1.0
        )
        features = self.extractor.extract_all(preprocessed)

        viz_payload: VisualizationPayload = self.builder.build_all(
            preprocessed=preprocessed,
            log_mel_result=features.log_mel_spectrogram
        )

        payload_dict = viz_payload.to_dict()
        
        # Verify JSON dumping succeeds without float non-serializable errors
        json_str = json.dumps(payload_dict)
        self.assertIsInstance(json_str, str)
        self.assertIn("waveform", payload_dict)
        self.assertIn("spectrogram", payload_dict)

    def test_empty_input_handling(self):
        empty_pcm = np.array([], dtype=np.float32)
        waveform = self.builder.build_waveform(empty_pcm, duration_seconds=0.0)

        self.assertEqual(waveform.target_points, 300)
        self.assertEqual(len(waveform.peak_envelope), 300)
        self.assertEqual(waveform.peak_envelope[0], 0.0)

        empty_log_mel = np.array([], dtype=np.float32)
        spectrogram = self.builder.build_spectrogram(empty_log_mel)

        self.assertEqual(spectrogram.freq_bins, 40)
        self.assertEqual(spectrogram.time_bins, 120)
        self.assertEqual(spectrogram.data_grid[0][0], 0.0)


if __name__ == "__main__":
    unittest.main()
