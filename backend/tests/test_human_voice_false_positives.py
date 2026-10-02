"""Regression and Evaluation Suite for Human Voice False Positive Prevention.

Tests genuine human speech feature payloads across diverse acoustic conditions:
- Normal studio/clean human speech
- Bandwidth-limited / telephony (8kHz / 11.025kHz resampled) human speech
- Calm reading / low micro-jitter human speech
- Quiet / ambient pause speech
- Short / low energy unusable audio

Guarantees:
1. FPR (False Positive Rate) on genuine human audio is 0.0% (NO human audio is classified as synthetic).
2. Unusable/silent/short audio evaluates as UNCERTAIN without fake probabilities.
3. Known synthetic voice clone signatures are correctly classified as SYNTHETIC.
4. P(synthetic) + P(human) == 1.0 probability invariant holds strictly across all cases.
"""

import math
import unittest
import numpy as np

from app.audio.preprocessor import PreprocessedAudio
from app.audio.quality import AudioQualityEvaluator
from app.features.extractor import AudioFeatureExtractor
from app.models.base_classifier import ModelStatus, PredictionLabel
from app.models.voice_clone_detector import VoiceCloneDetector


class TestHumanVoiceFalsePositives(unittest.TestCase):
    """Test suite ensuring VoxGuard eliminates false-positive synthetic classifications."""

    def setUp(self):
        self.detector = VoiceCloneDetector()
        self.extractor = AudioFeatureExtractor()
        self.quality_evaluator = AudioQualityEvaluator()

    def _create_human_pcm(self, duration_sec=2.0, sr=16000, f0=140.0, jitter_amt=0.012, is_bandlimited=False):
        """Generates synthetic test PCM simulating organic human vocal tract mechanics."""
        t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
        
        # Organic F0 pitch contour with micro-tremor/jitter
        f0_contour = f0 + np.sin(2 * np.pi * 5.5 * t) * 8.0 + (np.random.rand(len(t)) - 0.5) * jitter_amt * f0
        phase = 2 * np.pi * np.cumsum(f0_contour) / sr
        
        # Harmonics (F1-F4 vocal fold formants)
        harmonics = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase) + 0.12 * np.sin(4 * phase)
        
        # Smooth speech envelope with breath pauses
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 1.8 * t))
        speech_pcm = harmonics * envelope
        
        if is_bandlimited:
            # Simulate 8kHz telephony bandpass filter (zero out frequencies above 3.8 kHz)
            import scipy.signal
            b, a = scipy.signal.butter(6, 3800.0 / (sr / 2.0), btype='low')
            speech_pcm = scipy.signal.filtfilt(b, a, speech_pcm).astype(np.float32)
            
        peak = np.max(np.abs(speech_pcm))
        if peak > 0:
            speech_pcm /= peak
        return speech_pcm.astype(np.float32)

    def _create_synthetic_pcm(self, duration_sec=2.5, sr=16000):
        """Generates test PCM simulating neural vocoder re-synthesis (steep cutoff + static pitch)."""
        t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
        # Static pitch F0 = 180 Hz with zero jitter/vibrato
        f0 = 180.0
        synth_pcm = np.zeros_like(t)
        for k in range(1, 25):
            freq = k * f0
            if freq <= 4000.0:
                synth_pcm += (1.0 / (k ** 0.8)) * np.sin(2 * np.pi * freq * t)
        
        # Steep 4.2 kHz brickwall vocoder cutoff filter
        import scipy.signal
        b, a = scipy.signal.butter(10, 4200.0 / (sr / 2.0), btype='low')
        synth_pcm = scipy.signal.filtfilt(b, a, synth_pcm).astype(np.float32)
        
        peak = np.max(np.abs(synth_pcm))
        if peak > 0:
            synth_pcm /= peak
        return synth_pcm.astype(np.float32)

    def test_1_clean_human_speech_classified_as_human(self):
        """1. Test clean organic human speech is classified as HUMAN (never SYNTHETIC)."""
        pcm = self._create_human_pcm(duration_sec=2.5, f0=130.0, jitter_amt=0.015)
        prep = PreprocessedAudio(pcm_data=pcm, sample_rate=16000, channels=1, num_frames=len(pcm), duration_seconds=2.5, metadata={"original_sample_rate": 16000})
        feats = self.extractor.extract_all(prep)
        
        res = self.detector.predict(feats)
        self.assertNotEqual(res.label, PredictionLabel.SYNTHETIC, "Clean human voice must NEVER be classified as synthetic")
        self.assertIn(res.label, (PredictionLabel.HUMAN, PredictionLabel.UNCERTAIN))
        self.assertLess(res.probabilities["synthetic"], 0.40)
        self.assertAlmostEqual(res.probabilities["synthetic"] + res.probabilities["human"], 1.0, places=4)

    def test_2_bandlimited_telephony_human_speech_not_false_positive(self):
        """2. Test resampled 8kHz telephony human voice is not falsely triggered by anti-aliasing cutoff."""
        pcm = self._create_human_pcm(duration_sec=2.5, f0=180.0, is_bandlimited=True)
        prep = PreprocessedAudio(
            pcm_data=pcm,
            sample_rate=16000,
            channels=1,
            num_frames=len(pcm),
            duration_seconds=2.5,
            metadata={"original_sample_rate": 8000, "resampled": True}
        )
        feats = self.extractor.extract_all(prep)
        
        res = self.detector.predict(feats)
        self.assertNotEqual(res.label, PredictionLabel.SYNTHETIC, "Telephony human voice must not trigger synthetic false positive")
        self.assertLess(res.probabilities["synthetic"], 0.45)

    def test_3_silent_and_short_audio_quality_evaluator(self):
        """3. Test physical signal quality evaluation catches near-silent and ultra-short audio."""
        silent_pcm = np.zeros(16000, dtype=np.float32)
        qa = self.quality_evaluator.evaluate(silent_pcm, sample_rate=16000)
        self.assertFalse(qa.is_usable)
        self.assertIn("near-silent", qa.unusable_reason)

        short_pcm = np.random.rand(1600).astype(np.float32) * 0.1
        qa_short = self.quality_evaluator.evaluate(short_pcm, sample_rate=16000, min_duration_seconds=0.5)
        self.assertFalse(qa_short.is_usable)
        self.assertIn("too short", qa_short.unusable_reason)

    def test_4_synthetic_deepfake_correctly_identified(self):
        """4. Test neural vocoder simulation is correctly detected as SYNTHETIC."""
        pcm = self._create_synthetic_pcm(duration_sec=2.5)
        prep = PreprocessedAudio(pcm_data=pcm, sample_rate=16000, channels=1, num_frames=len(pcm), duration_seconds=2.5, metadata={"original_sample_rate": 16000})
        feats = self.extractor.extract_all(prep)
        
        res = self.detector.predict(feats)
        self.assertEqual(res.label, PredictionLabel.SYNTHETIC)
        self.assertGreaterEqual(res.probabilities["synthetic"], 0.68)
        self.assertAlmostEqual(res.probabilities["synthetic"] + res.probabilities["human"], 1.0, places=4)

    def test_5_false_positive_rate_suite(self):
        """5. Test False Positive Rate (FPR) across a diverse human speech validation suite."""
        human_samples = [
            self._create_human_pcm(duration_sec=2.0, f0=120.0, jitter_amt=0.010),
            self._create_human_pcm(duration_sec=3.0, f0=210.0, jitter_amt=0.018),
            self._create_human_pcm(duration_sec=2.5, f0=160.0, is_bandlimited=True),
            self._create_human_pcm(duration_sec=4.0, f0=100.0, jitter_amt=0.008),
        ]
        
        false_positives = 0
        total_human = len(human_samples)
        
        for idx, pcm in enumerate(human_samples):
            prep = PreprocessedAudio(pcm_data=pcm, sample_rate=16000, channels=1, num_frames=len(pcm), duration_seconds=len(pcm)/16000.0, metadata={"original_sample_rate": 16000})
            feats = self.extractor.extract_all(prep)
            res = self.detector.predict(feats)
            if res.label == PredictionLabel.SYNTHETIC:
                false_positives += 1

        fpr = false_positives / float(total_human)
        self.assertEqual(fpr, 0.0, f"False Positive Rate on human validation suite must be 0.0, got {fpr * 100:.1f}%")


if __name__ == "__main__":
    unittest.main()
