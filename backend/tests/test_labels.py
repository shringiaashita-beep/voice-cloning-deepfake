"""Unit tests for VoxGuard Label Encoding/Decoding module (app.evaluation.labels)."""

import unittest
from app.evaluation.labels import encode_label, decode_label, LABEL_TO_CODE, CODE_TO_LABEL


class TestLabels(unittest.TestCase):
    """Test suite for label encoding & decoding functions."""

    def test_encode_valid_labels(self):
        """Verify valid human and synthetic label encoding."""
        self.assertEqual(encode_label("human"), 0)
        self.assertEqual(encode_label("synthetic"), 1)
        self.assertEqual(encode_label(" HUMAN "), 0)
        self.assertEqual(encode_label("SYNTHETIC"), 1)

    def test_encode_invalid_label_raises_value_error(self):
        """Verify unknown or malformed label string raises ValueError."""
        with self.assertRaises(ValueError):
            encode_label("ai_generated")

        with self.assertRaises(ValueError):
            encode_label("fake")

        with self.assertRaises(ValueError):
            encode_label(123)  # Non-string input

    def test_decode_valid_codes(self):
        """Verify valid integer code decoding."""
        self.assertEqual(decode_label(0), "human")
        self.assertEqual(decode_label(1), "synthetic")

    def test_decode_invalid_code_raises_value_error(self):
        """Verify unknown code integer or invalid type raises ValueError."""
        with self.assertRaises(ValueError):
            decode_label(2)

        with self.assertRaises(ValueError):
            decode_label(-1)

        with self.assertRaises(ValueError):
            decode_label(True)  # Bool check

        with self.assertRaises(ValueError):
            decode_label("0")  # String code check


if __name__ == "__main__":
    unittest.main()
