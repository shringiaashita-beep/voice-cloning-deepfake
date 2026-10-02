"""Unit tests for VoxGuard Batch Collator (app.training.collator)."""

import unittest
import numpy as np

from app.training.collator import TrainingBatchCollator
from app.training.types import TrainingSample, TrainingBatch


class TestCollator(unittest.TestCase):
    """Test suite for TrainingBatchCollator component."""

    def setUp(self):
        self.collator = TrainingBatchCollator()
        self.sample1 = TrainingSample(
            log_mel=np.ones((80, 256), dtype=np.float32) * 1.0,
            mfcc=np.ones((20, 256), dtype=np.float32) * 1.0,
            spectral_centroid=np.ones((1, 256), dtype=np.float32) * 1.0,
            spectral_rolloff=np.ones((1, 256), dtype=np.float32) * 1.0,
            label=0,
            label_str="human",
            speaker_id="spk_HUMAN_01",
            source="libri",
            source_id="v1",
            sample_id="sample_001",
            window_index=0,
            split="train"
        )
        self.sample2 = TrainingSample(
            log_mel=np.ones((80, 256), dtype=np.float32) * 2.0,
            mfcc=np.ones((20, 256), dtype=np.float32) * 2.0,
            spectral_centroid=np.ones((1, 256), dtype=np.float32) * 2.0,
            spectral_rolloff=np.ones((1, 256), dtype=np.float32) * 2.0,
            label=1,
            label_str="synthetic",
            speaker_id="spk_SYNTH_101",
            source="eleven",
            source_id="v2",
            sample_id="sample_002",
            window_index=1,
            split="train"
        )

    def test_batch_collator_shapes(self):
        """Verify collating samples produces expected 3D and 1D batch shapes."""
        batch = self.collator.collate([self.sample1, self.sample2])

        self.assertIsInstance(batch, TrainingBatch)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.log_mel.shape, (2, 80, 256))
        self.assertEqual(batch.mfcc.shape, (2, 20, 256))
        self.assertEqual(batch.spectral_centroid.shape, (2, 1, 256))
        self.assertEqual(batch.spectral_rolloff.shape, (2, 1, 256))
        self.assertEqual(batch.labels.shape, (2,))

    def test_metadata_preservation(self):
        """Verify metadata (speaker_id, sample_id, source, window_index) is preserved."""
        batch = self.collator.collate([self.sample1, self.sample2])

        self.assertEqual(len(batch.metadata), 2)
        self.assertEqual(batch.metadata[0]["speaker_id"], "spk_HUMAN_01")
        self.assertEqual(batch.metadata[1]["speaker_id"], "spk_SYNTH_101")
        self.assertEqual(batch.metadata[0]["sample_id"], "sample_001")
        self.assertEqual(batch.metadata[1]["sample_id"], "sample_002")

    def test_speaker_id_preservation(self):
        """Verify speaker IDs remain accurately attached to their corresponding window tensors."""
        batch = self.collator.collate([self.sample1, self.sample2])

        # Batch index 0 array values are 1.0 -> spk_HUMAN_01
        self.assertEqual(np.mean(batch.log_mel[0]), 1.0)
        self.assertEqual(batch.metadata[0]["speaker_id"], "spk_HUMAN_01")

        # Batch index 1 array values are 2.0 -> spk_SYNTH_101
        self.assertEqual(np.mean(batch.log_mel[1]), 2.0)
        self.assertEqual(batch.metadata[1]["speaker_id"], "spk_SYNTH_101")

    def test_empty_collation_rejection(self):
        """Verify collating an empty list raises ValueError."""
        with self.assertRaises(ValueError):
            self.collator.collate([])


if __name__ == "__main__":
    unittest.main()
