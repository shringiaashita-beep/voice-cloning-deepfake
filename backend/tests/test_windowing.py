"""Unit tests for VoxGuard Feature Windowing (app.training.windowing)."""

import unittest
import numpy as np
from app.training.windowing import FeatureWindowing


class TestFeatureWindowing(unittest.TestCase):
    """Test suite for FeatureWindowing component."""

    def setUp(self):
        self.windower = FeatureWindowing(target_frames=256, stride=128)

    def test_short_sequence_padding(self):
        """Verify short sequence (T = 100 < 256) is padded to 256 with zeros."""
        t_short = 100
        log_mel = np.ones((80, t_short), dtype=np.float32)
        mfcc = np.ones((20, t_short), dtype=np.float32)
        centroid = np.ones((1, t_short), dtype=np.float32)
        rolloff = np.ones((1, t_short), dtype=np.float32)

        windows = self.windower.process_features(log_mel, mfcc, centroid, rolloff)

        self.assertEqual(len(windows), 1)
        win = windows[0]
        self.assertEqual(win["log_mel"].shape, (80, 256))
        self.assertEqual(win["mfcc"].shape, (20, 256))

        # Check right-side zero-padding
        self.assertTrue(np.all(win["log_mel"][:, :100] == 1.0))
        self.assertTrue(np.all(win["log_mel"][:, 100:] == 0.0))

    def test_exact_256_frame_input(self):
        """Verify sequence of exact 256 frames produces 1 window without padding or truncation."""
        log_mel = np.ones((80, 256), dtype=np.float32)
        mfcc = np.ones((20, 256), dtype=np.float32)
        centroid = np.ones((1, 256), dtype=np.float32)
        rolloff = np.ones((1, 256), dtype=np.float32)

        windows = self.windower.process_features(log_mel, mfcc, centroid, rolloff)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["log_mel"].shape, (80, 256))

    def test_long_sequence_windowing(self):
        """Verify long sequence (T = 600) generates multiple overlapping 256-frame windows with stride 128."""
        t_long = 600
        log_mel = np.arange(80 * t_long, dtype=np.float32).reshape(80, t_long)
        mfcc = np.ones((20, t_long), dtype=np.float32)
        centroid = np.ones((1, t_long), dtype=np.float32)
        rolloff = np.ones((1, t_long), dtype=np.float32)

        windows = self.windower.process_features(log_mel, mfcc, centroid, rolloff)

        # Expected slices: 0:256, 128:384, 256:512, 344:600 -> 4 windows
        self.assertEqual(len(windows), 4)

        for win in windows:
            self.assertEqual(win["log_mel"].shape, (80, 256))
            self.assertEqual(win["mfcc"].shape, (20, 256))

    def test_final_partial_window_handling(self):
        """Verify tail end window is right-aligned to cover remaining frames."""
        t_tail = 300
        log_mel = np.ones((80, t_tail), dtype=np.float32)
        mfcc = np.ones((20, t_tail), dtype=np.float32)
        centroid = np.ones((1, t_tail), dtype=np.float32)
        rolloff = np.ones((1, t_tail), dtype=np.float32)

        slices = self.windower.get_window_slices(t_tail)
        # 0..256, 44..300
        self.assertEqual(slices, [(0, 256), (44, 300)])

    def test_deterministic_windowing(self):
        """Verify processing the same array twice produces identical window tensors."""
        log_mel = np.random.randn(80, 500).astype(np.float32)
        mfcc = np.random.randn(20, 500).astype(np.float32)
        centroid = np.random.randn(1, 500).astype(np.float32)
        rolloff = np.random.randn(1, 500).astype(np.float32)

        res1 = self.windower.process_features(log_mel, mfcc, centroid, rolloff)
        res2 = self.windower.process_features(log_mel, mfcc, centroid, rolloff)

        self.assertEqual(len(res1), len(res2))
        for w1, w2 in zip(res1, res2):
            np.testing.assert_array_equal(w1["log_mel"], w2["log_mel"])


if __name__ == "__main__":
    unittest.main()
