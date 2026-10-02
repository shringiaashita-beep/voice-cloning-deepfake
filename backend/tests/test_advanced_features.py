"""Unit and Integration Tests for VoxGuard Advanced Features:
- URL Ingestion and SSRF Validation
- Speaker Biometrics & Reference Voice A/B Comparison
- Endpoints: POST /api/v1/analyze/url and POST /api/v1/compare
"""

import io
import unittest
import numpy as np

from app.audio.url_ingestor import AudioUrlIngestor, AudioUrlIngestionError
from app.forensics.speaker_comparator import SpeakerComparator
from app.features.extractor import AudioFeatures, FeatureResult
from app.models.base_classifier import ClassificationResult, ModelStatus, PredictionLabel
from tests.test_api_endpoints import generate_wav_bytes


class TestAdvancedFeatures(unittest.TestCase):
    """Test suite for URL Ingestion and Reference Voice Biometrics Comparison."""

    def setUp(self):
        self.ingestor = AudioUrlIngestor()
        self.comparator = SpeakerComparator()

    def test_url_ingestor_blocks_private_and_loopback_ips(self):
        """1. Verify SSRF safeguards reject loopback and private network targets."""
        with self.assertRaises(AudioUrlIngestionError):
            self.ingestor.fetch("http://127.0.0.1:8000/secret.wav")

        with self.assertRaises(AudioUrlIngestionError):
            self.ingestor.fetch("http://localhost:3000/test.mp3")

        with self.assertRaises(AudioUrlIngestionError):
            self.ingestor.fetch("ftp://example.com/audio.wav")

        with self.assertRaises(AudioUrlIngestionError):
            self.ingestor.fetch("not-a-valid-url")

        from app.audio.url_ingestor import SafeRedirectHandler
        handler = SafeRedirectHandler(self.ingestor._is_private_ip)
        with self.assertRaises(AudioUrlIngestionError):
            handler.redirect_request(None, None, 302, "Found", {}, "http://127.0.0.1/internal_secret.wav")
        with self.assertRaises(AudioUrlIngestionError):
            handler.redirect_request(None, None, 302, "Found", {}, "http://localhost:8080/audio.wav")
        with self.assertRaises(AudioUrlIngestionError):
            handler.redirect_request(None, None, 302, "Found", {}, "ftp://example.com/audio.wav")

    def test_speaker_biometrics_comparison_same_human(self):
        """2. Verify comparator on two authentic samples with close MFCC profiles."""
        # Create similar acoustic features
        mfcc_shared = np.random.normal(0.0, 1.0, (20, 100)).astype(np.float32)
        mfcc_a = mfcc_shared + np.random.normal(0.0, 0.05, (20, 100)).astype(np.float32)
        mfcc_b = mfcc_shared + np.random.normal(0.0, 0.05, (20, 100)).astype(np.float32)

        cent_a = np.full((1, 100), 1600.0, dtype=np.float32)
        cent_b = np.full((1, 100), 1620.0, dtype=np.float32)

        roll_a = np.full((1, 100), 5500.0, dtype=np.float32)
        roll_b = np.full((1, 100), 5550.0, dtype=np.float32)

        feat_a = AudioFeatures(
            log_mel_spectrogram=FeatureResult("mel", np.zeros((80, 100)), (80, 100), "float32", {}),
            mfcc=FeatureResult("mfcc", mfcc_a, (20, 100), "float32", {}),
            spectral_centroid=FeatureResult("cent", cent_a, (1, 100), "float32", {}),
            spectral_rolloff=FeatureResult("roll", roll_a, (1, 100), "float32", {}),
            sample_rate=16000,
            duration_seconds=2.0
        )
        feat_b = AudioFeatures(
            log_mel_spectrogram=FeatureResult("mel", np.zeros((80, 100)), (80, 100), "float32", {}),
            mfcc=FeatureResult("mfcc", mfcc_b, (20, 100), "float32", {}),
            spectral_centroid=FeatureResult("cent", cent_b, (1, 100), "float32", {}),
            spectral_rolloff=FeatureResult("roll", roll_b, (1, 100), "float32", {}),
            sample_rate=16000,
            duration_seconds=2.0
        )

        cls_a = ClassificationResult(
            label=PredictionLabel.HUMAN,
            probabilities={"human": 0.94, "synthetic": 0.06},
            confidence_score=0.94,
            model_name="VoiceCloneDetector",
            model_version="1.2.0",
            status=ModelStatus.READY,
            metadata={}
        )
        cls_b = ClassificationResult(
            label=PredictionLabel.HUMAN,
            probabilities={"human": 0.92, "synthetic": 0.08},
            confidence_score=0.92,
            model_name="VoiceCloneDetector",
            model_version="1.2.0",
            status=ModelStatus.READY,
            metadata={}
        )

        res = self.comparator.compare(feat_a, cls_a, feat_b, cls_b, "authentic_1.wav", "authentic_2.wav")
        self.assertEqual(res.impersonation_risk, "AUTHENTIC_MATCH")
        self.assertGreaterEqual(res.speaker_similarity_percent, 75.0)
        self.assertIn("Verified Same Human Speaker", res.verdict)

    def test_speaker_biometrics_comparison_cloned_impersonation(self):
        """3. Verify comparator detects clone impersonation when Sample A is synthetic and matches B."""
        mfcc_shared = np.random.normal(0.0, 1.0, (20, 100)).astype(np.float32)
        mfcc_a = mfcc_shared + np.random.normal(0.0, 0.04, (20, 100)).astype(np.float32)
        mfcc_b = mfcc_shared + np.random.normal(0.0, 0.04, (20, 100)).astype(np.float32)

        feat_a = AudioFeatures(
            log_mel_spectrogram=FeatureResult("mel", np.zeros((80, 100)), (80, 100), "float32", {}),
            mfcc=FeatureResult("mfcc", mfcc_a, (20, 100), "float32", {}),
            spectral_centroid=FeatureResult("cent", np.full((1, 100), 1500.0, dtype=np.float32), (1, 100), "float32", {}),
            spectral_rolloff=FeatureResult("roll", np.full((1, 100), 5000.0, dtype=np.float32), (1, 100), "float32", {}),
            sample_rate=16000,
            duration_seconds=2.0
        )
        feat_b = AudioFeatures(
            log_mel_spectrogram=FeatureResult("mel", np.zeros((80, 100)), (80, 100), "float32", {}),
            mfcc=FeatureResult("mfcc", mfcc_b, (20, 100), "float32", {}),
            spectral_centroid=FeatureResult("cent", np.full((1, 100), 1510.0, dtype=np.float32), (1, 100), "float32", {}),
            spectral_rolloff=FeatureResult("roll", np.full((1, 100), 5020.0, dtype=np.float32), (1, 100), "float32", {}),
            sample_rate=16000,
            duration_seconds=2.0
        )

        cls_synth = ClassificationResult(
            label=PredictionLabel.SYNTHETIC,
            probabilities={"human": 0.03, "synthetic": 0.97},
            confidence_score=0.97,
            model_name="VoiceCloneDetector",
            model_version="1.2.0",
            status=ModelStatus.READY,
            metadata={}
        )
        cls_human = ClassificationResult(
            label=PredictionLabel.HUMAN,
            probabilities={"human": 0.95, "synthetic": 0.05},
            confidence_score=0.95,
            model_name="VoiceCloneDetector",
            model_version="1.2.0",
            status=ModelStatus.READY,
            metadata={}
        )

        res = self.comparator.compare(feat_a, cls_synth, feat_b, cls_human, "questioned_clone.wav", "real_ceo.wav")
        self.assertEqual(res.impersonation_risk, "CRITICAL_IMPERSONATION")
        self.assertIn("Likely Voice Clone Impersonation", res.verdict)
        self.assertGreaterEqual(res.speaker_similarity_percent, 70.0)

    def test_compare_endpoint_execution(self):
        """4. Verify /compare endpoint accepts two files and returns comparison response."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        wav_a = generate_wav_bytes(duration_sec=2.0)
        wav_b = generate_wav_bytes(duration_sec=2.0)

        response = client.post(
            "/api/v1/compare?model=voice_clone_detector",
            files={
                "sample_a": ("sample_a.wav", io.BytesIO(wav_a), "audio/wav"),
                "sample_b": ("sample_b.wav", io.BytesIO(wav_b), "audio/wav")
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("comparison", data)
        self.assertIn("speaker_similarity_percent", data["comparison"])
        self.assertIn("impersonation_risk", data["comparison"])
        self.assertIn("sample_a_analysis", data)
        self.assertIn("sample_b_analysis", data)
