"""Unit tests for VoxGuard Training Configuration (app.training.config)."""

import unittest
from app.training.config import TrainingConfig


class TestTrainingConfig(unittest.TestCase):
    """Test suite for TrainingConfig dataclass."""

    def test_default_config_values(self):
        """Verify default configuration defaults match specification exactly."""
        config = TrainingConfig()
        self.assertEqual(config.TARGET_FRAMES, 256)
        self.assertEqual(config.WINDOW_STRIDE, 128)
        self.assertEqual(config.PADDING_MODE, "zero")
        self.assertEqual(config.TRUNCATION_MODE, "deterministic")
        self.assertEqual(config.RANDOM_SEED, 42)
        self.assertEqual(config.BATCH_SIZE, 16)
        self.assertTrue(config.SHUFFLE_TRAIN)
        self.assertFalse(config.SHUFFLE_VALIDATION)
        self.assertFalse(config.SHUFFLE_TEST)
        self.assertFalse(config.SHUFFLE_OOD)

    def test_custom_config_override(self):
        """Verify custom training parameters override default settings."""
        config = TrainingConfig(BATCH_SIZE=32, RANDOM_SEED=123, SHUFFLE_TRAIN=False)
        self.assertEqual(config.BATCH_SIZE, 32)
        self.assertEqual(config.RANDOM_SEED, 123)
        self.assertFalse(config.SHUFFLE_TRAIN)

    def test_config_to_dict(self):
        """Verify configuration to_dict dictionary serialization."""
        config = TrainingConfig()
        c_dict = config.to_dict()
        self.assertIsInstance(c_dict, dict)
        self.assertEqual(c_dict["TARGET_FRAMES"], 256)
        self.assertIn("log_mel", c_dict["FEATURES"])


if __name__ == "__main__":
    unittest.main()
