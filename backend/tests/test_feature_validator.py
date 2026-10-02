"""Unit tests for VoxGuard Feature Validator (app.evaluation.feature_validator)."""

import unittest
import numpy as np
from app.evaluation.feature_validator import FeatureValidator, FeatureValidationError


class TestFeatureValidator(unittest.TestCase):
    """Test suite for FeatureValidator tensor auditing."""

    def setUp(self):
        self.validator = FeatureValidator(expected_n_mels=80, expected_n_mfcc=20, expected_dtype="float32")
        self.num_frames = 50

        self.valid_log_mel = np.ones((80, self.num_frames), dtype=np.float32)
        self.valid_mfcc = np.ones((20, self.num_frames), dtype=np.float32)
        self.valid_centroid = np.ones((1, self.num_frames), dtype=np.float32)
        self.valid_rolloff = np.ones((1, self.num_frames), dtype=np.float32)

    def test_valid_features_pass(self):
        """Verify well-formed feature tensors pass validation cleanly."""
        res = self.validator.validate_features(
            log_mel=self.valid_log_mel,
            mfcc=self.valid_mfcc,
            spectral_centroid=self.valid_centroid,
            spectral_rolloff=self.valid_rolloff
        )
        self.assertTrue(res)

    def test_feature_dimensions_mismatch(self):
        """Verify incorrect feature axis-0 dimensions trigger FeatureValidationError."""
        bad_log_mel = np.ones((64, self.num_frames), dtype=np.float32)
        with self.assertRaises(FeatureValidationError):
            self.validator.validate_features(
                log_mel=bad_log_mel,
                mfcc=self.valid_mfcc,
                spectral_centroid=self.valid_centroid,
                spectral_rolloff=self.valid_rolloff
            )

        bad_mfcc = np.ones((13, self.num_frames), dtype=np.float32)
        with self.assertRaises(FeatureValidationError):
            self.validator.validate_features(
                log_mel=self.valid_log_mel,
                mfcc=bad_mfcc,
                spectral_centroid=self.valid_centroid,
                spectral_rolloff=self.valid_rolloff
            )

    def test_nan_detection(self):
        """Verify NaN values in any feature tensor trigger FeatureValidationError."""
        bad_mel = self.valid_log_mel.copy()
        bad_mel[10, 5] = np.nan
        with self.assertRaises(FeatureValidationError):
            self.validator.validate_features(
                log_mel=bad_mel,
                mfcc=self.valid_mfcc,
                spectral_centroid=self.valid_centroid,
                spectral_rolloff=self.valid_rolloff
            )

    def test_inf_detection(self):
        """Verify Infinity values in any feature tensor trigger FeatureValidationError."""
        bad_mfcc = self.valid_mfcc.copy()
        bad_mfcc[2, 3] = np.inf
        with self.assertRaises(FeatureValidationError):
            self.validator.validate_features(
                log_mel=self.valid_log_mel,
                mfcc=bad_mfcc,
                spectral_centroid=self.valid_centroid,
                spectral_rolloff=self.valid_rolloff
            )

    def test_dtype_validation(self):
        """Verify wrong numpy dtype (e.g. float64) triggers FeatureValidationError."""
        bad_dtype = self.valid_log_mel.astype(np.float64)
        with self.assertRaises(FeatureValidationError):
            self.validator.validate_features(
                log_mel=bad_dtype,
                mfcc=self.valid_mfcc,
                spectral_centroid=self.valid_centroid,
                spectral_rolloff=self.valid_rolloff
            )

    def test_temporal_frame_mismatch(self):
        """Verify mismatched temporal frame counts across features trigger FeatureValidationError."""
        mismatched_centroid = np.ones((1, self.num_frames + 10), dtype=np.float32)
        with self.assertRaises(FeatureValidationError):
            self.validator.validate_features(
                log_mel=self.valid_log_mel,
                mfcc=self.valid_mfcc,
                spectral_centroid=mismatched_centroid,
                spectral_rolloff=self.valid_rolloff
            )


if __name__ == "__main__":
    unittest.main()
