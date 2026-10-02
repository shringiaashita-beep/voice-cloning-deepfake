"""Comprehensive Test Suite for VoxGuard Phase 8 Model Infrastructure & Evaluation Readiness.

Tests model input/output contracts, manifests, validator service, evaluation result schemas,
speaker-disjoint dataset interfaces, calibration specs, registry endpoints, and safety fallbacks
without requiring external downloaded weights or datasets.
"""

import os
import unittest
import numpy as np

from app.evaluation.calibration import CalibrationConfig, CalibrationMethod, ProbabilityCalibrator
from app.evaluation.dataset_interface import AbstractAudioDataset, DatasetSample, DatasetSplit
from app.evaluation.evaluation_schema import BinaryClassificationMetrics, ConfusionMatrix, EvaluationReport
from app.features.extractor import AudioFeatureExtractor
from app.audio.preprocessor import PreprocessedAudio
from app.models.base_classifier import ClassificationResult, ModelStatus, PredictionLabel
from app.models.model_input import ModelInputContract
from app.models.model_manifest import ModelManifest
from app.models.model_output import ModelOutputContract
from app.models.model_validator import ModelValidator, ValidationResult
from app.models.ml_classifier import RealMLClassifier
from app.models.registry import ModelRegistry


class TestModelInfrastructurePhase8(unittest.TestCase):
    """Test suite for Phase 8 infrastructure abstractions and safety safeguards."""

    def setUp(self):
        """Create mock audio features for testing contracts."""
        extractor = AudioFeatureExtractor()
        t = np.linspace(0, 1.0, 16000, dtype=np.float32)
        sine_pcm = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        preprocessed = PreprocessedAudio(
            pcm_data=sine_pcm,
            sample_rate=16000,
            channels=1,
            num_frames=16000,
            duration_seconds=1.0,
            metadata={"dtype": "float32"}
        )
        self.features = extractor.extract_all(preprocessed)

    def test_1_valid_model_manifest(self):
        """1. Test valid ModelManifest creation, serialization, and deserialization."""
        manifest = ModelManifest(
            model_name="ResNet-LogMel Deepfake Net",
            model_version="1.2.0",
            architecture="resnet18",
            framework="torch",
            checkpoint_path="/tmp/fake.pt",
            checksum="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        self.assertEqual(manifest.model_name, "ResNet-LogMel Deepfake Net")
        self.assertEqual(manifest.expected_sample_rate, 16000)

        m_dict = manifest.to_dict()
        reconstructed = ModelManifest.from_dict(m_dict)
        self.assertEqual(reconstructed.model_version, "1.2.0")

    def test_2_invalid_manifest_detection(self):
        """2. Test ModelValidator detection of invalid manifest fields."""
        validator = ModelValidator()
        
        # Missing model_name
        inv_manifest = ModelManifest(model_name="", model_version="1.0.0")
        res = validator.validate_manifest(inv_manifest)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.error_code, "INVALID_MANIFEST_METADATA")

        # Invalid class labels
        inv_manifest2 = ModelManifest(model_name="Net", model_version="1.0.0", class_names=["unknown"])
        res2 = validator.validate_manifest(inv_manifest2)
        self.assertFalse(res2.is_valid)
        self.assertEqual(res2.error_code, "INVALID_CLASS_LABELS")

    def test_3_missing_checkpoint_file_handling(self):
        """3. Test ModelValidator detection of missing model checkpoint files."""
        validator = ModelValidator()
        res = validator.validate_checkpoint_file("/non_existent_dir/weights_123.onnx")
        self.assertFalse(res.is_valid)
        self.assertEqual(res.error_code, "FILE_NOT_FOUND")

    def test_4_unsupported_model_format(self):
        """4. Test ModelValidator rejection of unsupported model file extensions."""
        # Create a temp file with invalid extension
        temp_path = "test_temp_model.exe"
        with open(temp_path, "wb") as f:
            f.write(b"dummy_content_bytes")

        try:
            validator = ModelValidator()
            res = validator.validate_checkpoint_file(temp_path)
            self.assertFalse(res.is_valid)
            self.assertEqual(res.error_code, "UNSUPPORTED_FORMAT")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_5_checksum_mismatch_detection(self):
        """5. Test ModelValidator detection of SHA-256 checksum mismatches."""
        temp_path = "test_checksum_model.pt"
        with open(temp_path, "wb") as f:
            f.write(b"sample_model_weight_bytes_12345")

        try:
            validator = ModelValidator()
            # Wrong expected checksum
            wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"
            res = validator.validate_checkpoint_file(temp_path, expected_checksum=wrong_hash)
            self.assertFalse(res.is_valid)
            self.assertEqual(res.error_code, "CHECKSUM_MISMATCH")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_6_incompatible_input_shape_validation(self):
        """6. Test ModelInputContract validation against feature payloads."""
        contract = ModelInputContract(feature_type="log_mel_spectrogram", expected_freq_bins=80)
        is_valid, msg = contract.validate_features(self.features)
        self.assertTrue(is_valid)
        self.assertIsNone(msg)

        # Test tensor shape adapter output
        tensor_grid = contract.adapt_to_tensor_shape(self.features)
        self.assertEqual(tensor_grid.ndim, 4)  # (1, 1, 80, T)
        self.assertEqual(tensor_grid.shape[2], 80)

        # Incompatible freq bins contract
        mismatch_contract = ModelInputContract(feature_type="log_mel_spectrogram", expected_freq_bins=128)
        is_valid2, msg2 = mismatch_contract.validate_features(self.features)
        self.assertFalse(is_valid2)
        self.assertIn("expected 128", msg2)

    def test_7_probability_sum_validation_when_ready(self):
        """7. Test ModelOutputContract probability summation to ~1.0 when status is READY."""
        output = ModelOutputContract(
            label=PredictionLabel.HUMAN,
            human_probability=0.80,
            synthetic_probability=0.20,
            confidence_score=0.80,
            model_name="Test Model",
            model_version="1.0",
            model_type="pytorch",
            inference_time_ms=1.5,
            evaluation_status=ModelStatus.READY
        )
        out_dict = output.to_dict()
        p_h = out_dict["probabilities"]["human"]
        p_s = out_dict["probabilities"]["synthetic"]
        self.assertIsNotNone(p_h)
        self.assertIsNotNone(p_s)
        self.assertAlmostEqual(p_h + p_s, 1.0, places=3)

    def test_8_null_probability_enforcement_when_not_ready(self):
        """8 & 9. Test ModelOutputContract enforces None probabilities when NOT_READY."""
        output = ModelOutputContract(
            label=PredictionLabel.HUMAN,
            human_probability=0.95,          # Should be overwritten to None
            synthetic_probability=0.05,      # Should be overwritten to None
            confidence_score=0.95,
            model_name="Uninitialized Model",
            model_version="0.0",
            model_type="pytorch",
            inference_time_ms=0.0,
            evaluation_status=ModelStatus.NOT_READY
        )
        self.assertIsNone(output.human_probability)
        self.assertIsNone(output.synthetic_probability)
        self.assertIsNone(output.confidence_score)
        self.assertEqual(output.label, PredictionLabel.UNCERTAIN)

    def test_10_calibration_default_status(self):
        """10. Test ProbabilityCalibrator reports status='not_calibrated' by default."""
        cal_config = CalibrationConfig()
        self.assertEqual(cal_config.status, "not_calibrated")
        self.assertEqual(cal_config.method, CalibrationMethod.NONE)

        calibrator = ProbabilityCalibrator(cal_config)
        p_h, p_s = calibrator.calibrate(0.70, 0.30)
        self.assertAlmostEqual(p_h + p_s, 1.0, places=3)
        self.assertEqual(p_h, 0.70)

    def test_11_speaker_overlap_detection_in_dataset_interface(self):
        """12. Test dataset interface speaker-disjoint validation logic."""
        ds = AbstractAudioDataset(name="Test Benchmark Dataset")
        
        # Add train samples
        ds.add_sample(DatasetSample("s1", "path/1.wav", "human", speaker_id="spk_A", split=DatasetSplit.TRAIN))
        ds.add_sample(DatasetSample("s2", "path/2.wav", "synthetic", speaker_id="spk_B", split=DatasetSplit.TRAIN))

        # Add test samples with NO overlap
        ds.add_sample(DatasetSample("s3", "path/3.wav", "human", speaker_id="spk_C", split=DatasetSplit.TEST))
        is_disjoint, overlapping = ds.validate_speaker_disjointness()
        self.assertTrue(is_disjoint)
        self.assertEqual(len(overlapping), 0)

        # Add test sample with SPEAKER LEAKAGE (spk_A in TEST)
        ds.add_sample(DatasetSample("s4", "path/4.wav", "synthetic", speaker_id="spk_A", split=DatasetSplit.TEST))
        is_disjoint2, overlapping2 = ds.validate_speaker_disjointness()
        self.assertFalse(is_disjoint2)
        self.assertIn("spk_A", overlapping2)

    def test_12_ood_dataset_split_metadata_handling(self):
        """13. Test OOD (Out-Of-Distribution) dataset sample tagging."""
        sample = DatasetSample(
            sample_id="ood_1",
            audio_path="ood/sample.wav",
            label="synthetic",
            speaker_id="spk_OOD_1",
            split=DatasetSplit.OOD,
            domain_metadata={"generator": "ElevenLabs-v2", "codec": "mp3_128k"}
        )
        self.assertEqual(sample.split, DatasetSplit.OOD)
        self.assertEqual(sample.domain_metadata["generator"], "ElevenLabs-v2")

    def test_13_evaluation_report_schema(self):
        """Verify EvaluationReport schema instantiation without fake metric data."""
        metrics = BinaryClassificationMetrics(
            accuracy=None,
            precision=None,
            recall=None,
            f1_score=None,
            roc_auc=None,
            eer=None
        )
        cm = ConfusionMatrix(tp=0, fp=0, tn=0, fn=0)
        report = EvaluationReport(
            dataset_name="ASVspoof 2019 LA Test",
            protocol_name="zero_speaker_overlap",
            speaker_disjoint_verified=True,
            metrics=metrics,
            confusion_matrix=cm,
            notes="Evaluated schema initialized without fake performance data."
        )

        r_dict = report.to_dict()
        self.assertEqual(r_dict["dataset_name"], "ASVspoof 2019 LA Test")
        self.assertTrue(r_dict["speaker_disjoint_verified"])
        self.assertIsNone(r_dict["metrics"]["accuracy"])

    def test_15_probability_invariant_and_screenshot_scenario(self):
        """16. Test probability invariant validation, NaN/Inf checks, and screenshot scenario contract."""
        # A) Invariant validation: synthetic + human == 1.0
        res = ClassificationResult(
            label=PredictionLabel.SYNTHETIC,
            probabilities={"synthetic": 0.885, "human": 0.115},
            confidence_score=0.885,
            model_name="VoxGuard Neural Acoustic Ensemble",
            model_version="1.2.0",
            status=ModelStatus.READY,
            metadata={"human_vocal_trait_similarity": 0.984}
        )
        self.assertEqual(res.label, PredictionLabel.SYNTHETIC)
        self.assertAlmostEqual(res.probabilities["synthetic"] + res.probabilities["human"], 1.0, places=5)
        self.assertAlmostEqual(res.probabilities["synthetic"], 0.885)
        self.assertAlmostEqual(res.probabilities["human"], 0.115)

        # B) Rejection of NaN / Infinity
        with self.assertRaises(ValueError):
            ClassificationResult(
                label=PredictionLabel.SYNTHETIC,
                probabilities={"synthetic": float('nan'), "human": 0.115},
                confidence_score=0.885,
                model_name="Test Model",
                model_version="1.0",
                status=ModelStatus.READY,
                metadata={}
            )

        # C) Label agreement enforcement
        res_mismatch = ClassificationResult(
            label=PredictionLabel.HUMAN,  # Mismatched label
            probabilities={"synthetic": 0.885, "human": 0.115},
            confidence_score=0.885,
            model_name="Test Model",
            model_version="1.0",
            status=ModelStatus.READY,
            metadata={}
        )
        # Should be auto-corrected to SYNTHETIC
        self.assertEqual(res_mismatch.label, PredictionLabel.SYNTHETIC)


if __name__ == "__main__":
    unittest.main()

