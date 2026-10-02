"""Unit and Integration Tests for VoxGuard Voice Clone Forensics Neural Ensemble Detector."""

import math
import unittest
import numpy as np

from app.audio.preprocessor import PreprocessedAudio
from app.features.extractor import AudioFeatureExtractor, AudioFeatures, FeatureResult
from app.models.base_classifier import ModelStatus, PredictionLabel
from app.models.registry import model_registry
from app.models.voice_clone_detector import VoiceCloneDetector
from app.routes.analysis import analyze_audio_stream
from tests.test_api_endpoints import generate_wav_bytes


class TestVoiceCloneDetector(unittest.TestCase):
    """Test suite for VoiceCloneDetector and voice clone forensics pipeline."""

    def setUp(self):
        self.detector = VoiceCloneDetector()

    def test_detector_metadata_and_readiness(self):
        """1. Verify VoiceCloneDetector metadata, version, and READY status."""
        self.assertEqual(self.detector.status, ModelStatus.READY)
        self.assertTrue(self.detector.is_ready())
        self.assertIn("Voice Clone", self.detector.model_name)

        meta = self.detector.get_model_metadata()
        self.assertEqual(meta.framework, "neural_acoustic_ensemble")
        self.assertIsNotNone(meta.evaluation_metrics)
        self.assertIn("eer_percent", meta.evaluation_metrics)

    def test_predict_on_synthetic_profile(self):
        """2. Verify detection on simulated neural vocoder features (steep high-band cutoff, flat jitter)."""
        # Create log_mel with steep high-frequency attenuation (typical of neural vocoders)
        log_mel = np.zeros((80, 200), dtype=np.float32)
        log_mel[:55, :] = -20.0  # active speech in low/mid bands
        log_mel[55:, :] = -78.0  # sharp cutoff in high frequencies (above 5.5 kHz)

        # MFCC with unnaturally smooth high coefficients
        mfcc = np.zeros((20, 200), dtype=np.float32)
        mfcc[:10, :] = np.random.normal(0, 1.0, (10, 200)).astype(np.float32)
        mfcc[10:, :] = np.random.normal(0, 0.04, (10, 200)).astype(np.float32)

        # Extremely flat spectral centroid (near zero micro-jitter)
        centroid = np.full((1, 200), 1200.0, dtype=np.float32)
        rolloff = np.full((1, 200), 4500.0, dtype=np.float32)

        features = AudioFeatures(
            log_mel_spectrogram=FeatureResult("log_mel", log_mel, log_mel.shape, "float32", {}),
            mfcc=FeatureResult("mfcc", mfcc, mfcc.shape, "float32", {}),
            spectral_centroid=FeatureResult("spectral_centroid", centroid, centroid.shape, "float32", {}),
            spectral_rolloff=FeatureResult("spectral_rolloff", rolloff, rolloff.shape, "float32", {}),
            sample_rate=16000,
            duration_seconds=2.0
        )

        result = self.detector.predict(features)
        self.assertEqual(result.status, ModelStatus.READY)
        self.assertEqual(result.label, PredictionLabel.SYNTHETIC)
        self.assertIsNotNone(result.probabilities)
        self.assertGreater(result.probabilities["synthetic"], 0.70)
        self.assertTrue(math.isclose(result.probabilities["human"] + result.probabilities["synthetic"], 1.0, abs_tol=0.01))

        # Check architecture fingerprint scores
        self.assertIn("architecture_scores", result.metadata)
        self.assertIn("elevenlabs_neural_vocoder", result.metadata["architecture_scores"])
        self.assertIn("voice_clone_indicators", result.metadata)
        self.assertGreater(len(result.metadata["voice_clone_indicators"]), 0)

    def test_predict_on_human_profile(self):
        """3. Verify detection on simulated organic human features (natural jitter, rich MFCC variance)."""
        # Natural speech across all bands
        log_mel = np.random.normal(-35.0, 12.0, (80, 200)).astype(np.float32)
        # Higher bands have natural acoustic energy
        log_mel[55:, :] = np.random.normal(-45.0, 10.0, (25, 200)).astype(np.float32)

        # MFCC with rich vocal tract variance across all 20 coefficients
        mfcc = np.random.normal(0.0, 0.8, (20, 200)).astype(np.float32)

        # Natural organic jitter variations in centroid
        t = np.linspace(0, 2.0, 200)
        centroid = (1500.0 + 300.0 * np.sin(2 * np.pi * 3.5 * t) + np.random.normal(0, 70.0, 200)).astype(np.float32)
        centroid = centroid.reshape(1, -1)
        rolloff = np.full((1, 200), 6500.0, dtype=np.float32)

        features = AudioFeatures(
            log_mel_spectrogram=FeatureResult("log_mel", log_mel, log_mel.shape, "float32", {}),
            mfcc=FeatureResult("mfcc", mfcc, mfcc.shape, "float32", {}),
            spectral_centroid=FeatureResult("spectral_centroid", centroid, centroid.shape, "float32", {}),
            spectral_rolloff=FeatureResult("spectral_rolloff", rolloff, rolloff.shape, "float32", {}),
            sample_rate=16000,
            duration_seconds=2.0
        )

        result = self.detector.predict(features)
        self.assertEqual(result.status, ModelStatus.READY)
        self.assertIsNotNone(result.probabilities)
        self.assertGreater(result.probabilities["human"], 0.60)
        self.assertEqual(result.label, PredictionLabel.HUMAN)

    def test_analyze_audio_stream_with_voice_clone_detector(self):
        """4. Verify end-to-end analyze_audio_stream when requesting model_id='voice_clone_detector'."""
        wav_bytes = generate_wav_bytes(duration_sec=3.0, sample_rate=16000)
        res = analyze_audio_stream(wav_bytes, filename="sample.wav", content_type="audio/wav", model_id="voice_clone_detector")

        self.assertEqual(res["http_status"], 200)
        body = res["response"]
        cls = body["classification"]
        self.assertEqual(cls["status"], "ready")
        self.assertIsNotNone(cls["probabilities"]["human"])
        self.assertIsNotNone(cls["probabilities"]["synthetic"])
        self.assertTrue(math.isclose(cls["probabilities"]["human"] + cls["probabilities"]["synthetic"], 1.0, abs_tol=0.01))

        # Forensic report check
        report = body["forensic_report"]
        self.assertIsNotNone(report["model_assessment"])
        self.assertIn("evidence_indicators", report)
        # Should include ML model indicators
        ml_indicators = [ind for ind in report["evidence_indicators"] if ind["provenance"] == "ml_model"]
        self.assertGreater(len(ml_indicators), 0)

    def test_model_registry_contains_voice_clone_detector(self):
        """5. Verify ModelRegistry includes 'voice_clone_detector' and can switch models."""
        models = model_registry.list_models()
        self.assertIn("voice_clone_detector", models)
        self.assertEqual(models["voice_clone_detector"]["status"], "ready")

        detector_instance = model_registry.get_model("voice_clone_detector")
        self.assertIsInstance(detector_instance, VoiceCloneDetector)

    def test_predict_on_microphone_recorded_human_speech(self):
        """6. Verify genuine human voice recorded via standard consumer microphone is classified as HUMAN."""
        # Realistic microphone capture: high bands naturally attenuate ~20-25 dB below mid speech bands
        log_mel = np.zeros((80, 200), dtype=np.float32)
        log_mel[:55, :] = np.random.normal(-24.0, 6.0, (55, 200)).astype(np.float32)
        log_mel[55:, :] = np.random.normal(-45.0, 5.0, (25, 200)).astype(np.float32)  # -21 dB natural roll-off

        # Biological vocal tract resonances (rich variance in MFCC 10-20)
        mfcc = np.zeros((20, 200), dtype=np.float32)
        mfcc[:10, :] = np.random.normal(0, 1.2, (10, 200)).astype(np.float32)
        mfcc[10:, :] = np.random.normal(0, 0.45, (10, 200)).astype(np.float32)

        # Natural vocal fold micro-jitter (~20% micro-jitter index, identical to user's recording)
        t = np.linspace(0, 2.0, 200)
        centroid = (1600.0 + 250.0 * np.sin(2 * np.pi * 4.0 * t) + np.random.normal(0, 80.0, 200)).astype(np.float32)
        centroid = centroid.reshape(1, -1)
        rolloff = np.full((1, 200), 5500.0, dtype=np.float32)

        features = AudioFeatures(
            log_mel_spectrogram=FeatureResult("log_mel", log_mel, log_mel.shape, "float32", {}),
            mfcc=FeatureResult("mfcc", mfcc, mfcc.shape, "float32", {}),
            spectral_centroid=FeatureResult("spectral_centroid", centroid, centroid.shape, "float32", {}),
            spectral_rolloff=FeatureResult("spectral_rolloff", rolloff, rolloff.shape, "float32", {}),
            sample_rate=16000,
            duration_seconds=2.0
        )

        result = self.detector.predict(features)
        self.assertEqual(result.status, ModelStatus.READY)
        self.assertEqual(result.label, PredictionLabel.HUMAN)
        self.assertIsNotNone(result.probabilities)
        self.assertGreater(result.probabilities["human"], 0.75)
        self.assertLess(result.probabilities["synthetic"], 0.25)
        self.assertEqual(result.metadata["synthetic_risk_level"], "GENUINE_HUMAN")
        self.assertIn("Natural Biological Vocal Tract", result.metadata["voice_clone_architecture"])

