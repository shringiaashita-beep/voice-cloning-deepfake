"""Unit tests for VoxGuard Training Batch & Tensor Validator (app.training.validation)."""

import unittest
import numpy as np

from app.training.types import TrainingBatch, TrainingSample
from app.training.validation import TrainingBatchValidator, TrainingValidationError


class TestTrainingValidation(unittest.TestCase):
    """Test suite for TrainingBatchValidator tensor and batch auditing."""

    def setUp(self):
        self.validator = TrainingBatchValidator(target_frames=256)

        self.valid_sample = TrainingSample(
            log_mel=np.ones((80, 256), dtype=np.float32),
            mfcc=np.ones((20, 256), dtype=np.float32),
            spectral_centroid=np.ones((1, 256), dtype=np.float32),
            spectral_rolloff=np.ones((1, 256), dtype=np.float32),
            label=0,
            label_str="human",
            speaker_id="spk_001",
            source="libri",
            source_id="v1",
            sample_id="sample_001",
            window_index=0,
            split="train"
        )

    def test_sample_validation_pass(self):
        """Verify well-formed TrainingSample passes validation."""
        self.assertTrue(self.validator.validate_sample(self.valid_sample))

    def test_sample_missing_speaker_id(self):
        """Verify missing speaker_id raises TrainingValidationError."""
        bad_sample = TrainingSample(
            log_mel=self.valid_sample.log_mel,
            mfcc=self.valid_sample.mfcc,
            spectral_centroid=self.valid_sample.spectral_centroid,
            spectral_rolloff=self.valid_sample.spectral_rolloff,
            label=0,
            label_str="human",
            speaker_id="",  # Missing speaker ID
            source="libri",
            source_id="v1",
            sample_id="sample_001",
            window_index=0,
            split="train"
        )
        with self.assertRaises(TrainingValidationError):
            self.validator.validate_sample(bad_sample)

    def test_sample_invalid_label_integer(self):
        """Verify label integer code != 0 or 1 raises TrainingValidationError."""
        bad_sample = TrainingSample(
            log_mel=self.valid_sample.log_mel,
            mfcc=self.valid_sample.mfcc,
            spectral_centroid=self.valid_sample.spectral_centroid,
            spectral_rolloff=self.valid_sample.spectral_rolloff,
            label=5,  # Invalid label
            label_str="invalid",
            speaker_id="spk_01",
            source="src",
            source_id="v1",
            sample_id="sample_001",
            window_index=0,
            split="train"
        )
        with self.assertRaises(TrainingValidationError):
            self.validator.validate_sample(bad_sample)

    def test_sample_nan_detection(self):
        """Verify NaN tensor values raise TrainingValidationError."""
        bad_mel = self.valid_sample.log_mel.copy()
        bad_mel[10, 50] = np.nan
        bad_sample = TrainingSample(
            log_mel=bad_mel,
            mfcc=self.valid_sample.mfcc,
            spectral_centroid=self.valid_sample.spectral_centroid,
            spectral_rolloff=self.valid_sample.spectral_rolloff,
            label=0,
            label_str="human",
            speaker_id="spk_01",
            source="src",
            source_id="v1",
            sample_id="s1",
            window_index=0,
            split="train"
        )
        with self.assertRaises(TrainingValidationError):
            self.validator.validate_sample(bad_sample)

    def test_sample_inf_detection(self):
        """Verify Infinity tensor values raise TrainingValidationError."""
        bad_mfcc = self.valid_sample.mfcc.copy()
        bad_mfcc[5, 10] = np.inf
        bad_sample = TrainingSample(
            log_mel=self.valid_sample.log_mel,
            mfcc=bad_mfcc,
            spectral_centroid=self.valid_sample.spectral_centroid,
            spectral_rolloff=self.valid_sample.spectral_rolloff,
            label=0,
            label_str="human",
            speaker_id="spk_01",
            source="src",
            source_id="v1",
            sample_id="s1",
            window_index=0,
            split="train"
        )
        with self.assertRaises(TrainingValidationError):
            self.validator.validate_sample(bad_sample)

    def test_sample_wrong_dtype(self):
        """Verify float64 array raises TrainingValidationError."""
        bad_mel = self.valid_sample.log_mel.astype(np.float64)
        bad_sample = TrainingSample(
            log_mel=bad_mel,
            mfcc=self.valid_sample.mfcc,
            spectral_centroid=self.valid_sample.spectral_centroid,
            spectral_rolloff=self.valid_sample.spectral_rolloff,
            label=0,
            label_str="human",
            speaker_id="spk_01",
            source="src",
            source_id="v1",
            sample_id="s1",
            window_index=0,
            split="train"
        )
        with self.assertRaises(TrainingValidationError):
            self.validator.validate_sample(bad_sample)

    def test_batch_validation_pass(self):
        """Verify valid TrainingBatch passes validation."""
        b_size = 2
        batch = TrainingBatch(
            log_mel=np.ones((b_size, 80, 256), dtype=np.float32),
            mfcc=np.ones((b_size, 20, 256), dtype=np.float32),
            spectral_centroid=np.ones((b_size, 1, 256), dtype=np.float32),
            spectral_rolloff=np.ones((b_size, 1, 256), dtype=np.float32),
            labels=np.array([0, 1], dtype=np.int64),
            metadata=[
                {"sample_id": "s1", "speaker_id": "spk1"},
                {"sample_id": "s2", "speaker_id": "spk2"}
            ]
        )
        self.assertTrue(self.validator.validate_batch(batch))

    def test_batch_shape_mismatch(self):
        """Verify incorrect batch tensor shape raises TrainingValidationError."""
        batch = TrainingBatch(
            log_mel=np.ones((2, 64, 256), dtype=np.float32),  # Wrong axis-1 dim (64 instead of 80)
            mfcc=np.ones((2, 20, 256), dtype=np.float32),
            spectral_centroid=np.ones((2, 1, 256), dtype=np.float32),
            spectral_rolloff=np.ones((2, 1, 256), dtype=np.float32),
            labels=np.array([0, 1], dtype=np.int64),
            metadata=[
                {"sample_id": "s1", "speaker_id": "spk1"},
                {"sample_id": "s2", "speaker_id": "spk2"}
            ]
        )
        with self.assertRaises(TrainingValidationError):
            self.validator.validate_batch(batch)


if __name__ == "__main__":
    unittest.main()
