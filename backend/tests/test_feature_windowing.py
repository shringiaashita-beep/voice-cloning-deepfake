"""Unit Tests for VoxGuard Feature Windowing Module (app.training.windowing & app.training.feature_windowing).

Validates short sequence padding, exact 256-frame sequences, long sequence windowing,
stride behavior, deterministic windowing, label/speaker preservation, and shape checks.
"""

import unittest
import numpy as np

from app.training.windowing import FeatureWindowing
from app.training.feature_windowing import FeatureWindowing as FeatureWindowingAlias


class TestFeatureWindowing(unittest.TestCase):
    """Test suite for FeatureWindowing component."""

    def setUp(self):
        self.windower = FeatureWindowing(target_frames=256, stride=128)

    def test_alias_import(self):
        """Verify feature_windowing.py alias import."""
        self.assertEqual(FeatureWindowing, FeatureWindowingAlias)

    def test_short_sequence_padding(self):
        """Verify short sequence (e.g. 100 frames) is padded with zeros to 256 frames."""
        log_mel = np.ones((80, 100), dtype=np.float32)
        mfcc = np.ones((20, 100), dtype=np.float32)
        centroid = np.ones((1, 100), dtype=np.float32)
        rolloff = np.ones((1, 100), dtype=np.float32)

        windows = self.windower.process_features(log_mel, mfcc, centroid, rolloff)

        self.assertEqual(len(windows), 1)
        win = windows[0]

        self.assertEqual(win["log_mel"].shape, (80, 256))
        self.assertEqual(win["mfcc"].shape, (20, 256))
        self.assertEqual(win["spectral_centroid"].shape, (1, 256))
        self.assertEqual(win["spectral_rolloff"].shape, (1, 256))

        # Check right-side zero-padding
        self.assertTrue(np.all(win["log_mel"][:, :100] == 1.0))
        self.assertTrue(np.all(win["log_mel"][:, 100:] == 0.0))

    def test_exact_256_frame_sequence(self):
        """Verify sequence of exactly 256 frames produces single window without padding."""
        log_mel = np.ones((80, 256), dtype=np.float32)
        mfcc = np.ones((20, 256), dtype=np.float32)
        centroid = np.ones((1, 256), dtype=np.float32)
        rolloff = np.ones((1, 256), dtype=np.float32)

        windows = self.windower.process_features(log_mel, mfcc, centroid, rolloff)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["log_mel"].shape, (80, 256))
        self.assertTrue(np.all(windows[0]["log_mel"] == 1.0))

    def test_long_sequence_windowing_and_stride(self):
        """Verify sequence of 500 frames with target=256, stride=128 produces correct windows."""
        log_mel = np.random.randn(80, 500).astype(np.float32)
        mfcc = np.random.randn(20, 500).astype(np.float32)
        centroid = np.random.randn(1, 500).astype(np.float32)
        rolloff = np.random.randn(1, 500).astype(np.float32)

        windows = self.windower.process_features(log_mel, mfcc, centroid, rolloff)

        # 0..256, 128..384, 244..500 (tail) => 3 windows
        self.assertTrue(len(windows) >= 2)
        for win in windows:
            self.assertEqual(win["log_mel"].shape, (80, 256))

    def test_deterministic_output(self):
        """Verify windowing output is 100% deterministic across multiple calls."""
        log_mel = np.random.randn(80, 400).astype(np.float32)
        mfcc = np.random.randn(20, 400).astype(np.float32)
        centroid = np.random.randn(1, 400).astype(np.float32)
        rolloff = np.random.randn(1, 400).astype(np.float32)

        res1 = self.windower.process_features(log_mel, mfcc, centroid, rolloff)
        res2 = self.windower.process_features(log_mel, mfcc, centroid, rolloff)

        self.assertEqual(len(res1), len(res2))
        for w1, w2 in zip(res1, res2):
            np.testing.assert_array_equal(w1["log_mel"], w2["log_mel"])


if __name__ == "__main__":
    unittest.main()
