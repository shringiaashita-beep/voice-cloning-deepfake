"""Unit tests for VoxGuard Audio Validator Service.

Tests valid formats, invalid formats, corrupted inputs, file size limits,
duration limits, boundary conditions, and path traversal protection.
Does not depend on external datasets or model weights.
"""

import io
import math
import struct
import unittest
import wave

from app.audio.validator import AudioErrorCode, AudioValidator
from app.config import Settings


# Helper functions to synthesize minimal in-memory audio payloads for testing

def generate_wav_bytes(duration_sec: float, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Generates valid in-memory PCM WAV audio data."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)
        num_frames = int(duration_sec * sample_rate)
        # Generate simple silence / zero samples
        data = b"\x00\x00" * channels * num_frames
        wav_file.writeframesraw(data)
    return buf.getvalue()


def generate_flac_bytes(duration_sec: float, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Generates valid in-memory FLAC header payload."""
    header = b"fLaC"
    block_header = b"\x00\x00\x00\x22"  # 34-byte STREAMINFO block
    blocks = b"\x10\x00\x10\x00" + b"\x00\x00\x00\x00\x00\x00"
    
    total_samples = int(duration_sec * sample_rate)
    sr_chan_bits_samples = (sample_rate << 44) | ((channels - 1) << 41) | (15 << 36) | (total_samples & 0xFFFFFFFFF)
    sr_chan_bytes = sr_chan_bits_samples.to_bytes(8, byteorder="big")
    md5_hash = b"\x00" * 16
    
    return header + block_header + blocks + sr_chan_bytes + md5_hash


def generate_mp3_bytes(duration_sec: float = 2.0) -> bytes:
    """Generates valid synthetic MP3 payload with ID3v2 tag and frame header."""
    id3_tag = b"ID3\x04\x00\x00\x00\x00\x00\x0aTITLE\x00\x00\x00"
    frame_header = b"\xff\xfb\x90\x00"  # MPEG-1, Layer III, 44100 Hz, stereo
    payload = b"\x00" * int(duration_sec * 44100 * 2 * 0.5)
    return id3_tag + frame_header + payload


def generate_ogg_bytes(duration_sec: float = 2.0) -> bytes:
    """Generates valid synthetic OGG payload with OpusHead tag."""
    header = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00"
    body = b"OpusHead\x01\x01\x38\x01\x80\xbb\x00\x00\x00\x00\x00"
    payload = b"\x00" * int(duration_sec * 16000 * 2)
    return header + body + payload


# --- Test Suite ---

class TestAudioValidatorValidCases(unittest.TestCase):
    """Valid audio format test cases."""

    def test_valid_wav(self):
        validator = AudioValidator()
        wav_data = generate_wav_bytes(duration_sec=2.5, sample_rate=16000, channels=1)
        result = validator.validate(wav_data, filename="sample.wav", content_type="audio/wav")

        self.assertTrue(result.is_valid)
        self.assertEqual(result.format, "wav")
        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(result.channels, 1)
        self.assertTrue(math.isclose(result.duration_seconds, 2.5, abs_tol=0.05))
        self.assertIsNone(result.error_code)

    def test_valid_mp3(self):
        validator = AudioValidator()
        mp3_data = generate_mp3_bytes(duration_sec=2.0)
        result = validator.validate(mp3_data, filename="sample.mp3", content_type="audio/mpeg")

        self.assertTrue(result.is_valid)
        self.assertEqual(result.format, "mp3")
        self.assertIsNone(result.error_code)

    def test_valid_flac(self):
        validator = AudioValidator()
        flac_data = generate_flac_bytes(duration_sec=3.0, sample_rate=44100, channels=2)
        result = validator.validate(flac_data, filename="sample.flac", content_type="audio/flac")

        self.assertTrue(result.is_valid)
        self.assertEqual(result.format, "flac")
        self.assertEqual(result.sample_rate, 44100)
        self.assertEqual(result.channels, 2)
        self.assertTrue(math.isclose(result.duration_seconds, 3.0, abs_tol=0.05))

    def test_valid_ogg(self):
        validator = AudioValidator()
        ogg_data = generate_ogg_bytes(duration_sec=2.0)
        result = validator.validate(ogg_data, filename="sample.ogg", content_type="audio/ogg")

        self.assertTrue(result.is_valid)
        self.assertEqual(result.format, "ogg")


class TestAudioValidatorInvalidCases(unittest.TestCase):
    """Invalid and malformed input test cases."""

    def test_missing_input(self):
        validator = AudioValidator()
        result = validator.validate(None)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.error_code, AudioErrorCode.MISSING_INPUT.value)

    def test_empty_file(self):
        validator = AudioValidator()
        result = validator.validate(b"", filename="empty.wav")

        self.assertFalse(result.is_valid)
        self.assertEqual(result.error_code, AudioErrorCode.EMPTY_FILE.value)

    def test_unsupported_extension(self):
        validator = AudioValidator()
        wav_data = generate_wav_bytes(duration_sec=1.0)
        result = validator.validate(wav_data, filename="sample.exe", content_type="application/octet-stream")

        self.assertFalse(result.is_valid)
        self.assertEqual(result.error_code, AudioErrorCode.UNSUPPORTED_EXTENSION.value)

    def test_unsupported_mime_type(self):
        validator = AudioValidator()
        wav_data = generate_wav_bytes(duration_sec=1.0)
        result = validator.validate(wav_data, filename="sample.wav", content_type="application/pdf")

        self.assertFalse(result.is_valid)
        self.assertEqual(result.error_code, AudioErrorCode.UNSUPPORTED_MIME_TYPE.value)

    def test_corrupted_audio(self):
        validator = AudioValidator()
        garbage_data = b"RIFF" + b"x" * 100  # Invalid WAV structure
        result = validator.validate(garbage_data, filename="corrupt.wav")

        self.assertFalse(result.is_valid)
        self.assertEqual(result.error_code, AudioErrorCode.CORRUPTED_AUDIO.value)

    def test_text_file_renamed_to_wav(self):
        validator = AudioValidator()
        text_data = b"Hello, this is a plain text file pretending to be audio."
        result = validator.validate(text_data, filename="fake.wav", content_type="audio/wav")

        self.assertFalse(result.is_valid)
        self.assertEqual(result.error_code, AudioErrorCode.CORRUPTED_AUDIO.value)

    def test_audio_exceeding_max_duration(self):
        custom_settings = Settings(MAX_AUDIO_DURATION_SECONDS=5.0)
        validator = AudioValidator(config=custom_settings)
        long_wav = generate_wav_bytes(duration_sec=10.0)
        result = validator.validate(long_wav, filename="long.wav")

        self.assertFalse(result.is_valid)
        self.assertEqual(result.error_code, AudioErrorCode.AUDIO_DURATION_EXCEEDED.value)
        self.assertIn("exceeds maximum allowed limit", result.error_message)


class TestAudioValidatorBoundaryCases(unittest.TestCase):
    """Boundary file size and duration test cases."""

    def test_file_exactly_at_size_limit(self):
        custom_settings = Settings(MAX_UPLOAD_SIZE_MB=1)
        validator = AudioValidator(config=custom_settings)
        target_size = 1048576  # 1 MB
        base_wav = generate_wav_bytes(duration_sec=1.0)
        padding = b"\x00" * (target_size - len(base_wav))
        exact_bytes = base_wav + padding

        result = validator.validate(exact_bytes, filename="exact.wav")
        self.assertEqual(result.size_bytes, target_size)
        self.assertTrue(result.is_valid)

    def test_file_just_above_size_limit(self):
        custom_settings = Settings(MAX_UPLOAD_SIZE_MB=1)
        validator = AudioValidator(config=custom_settings)
        target_size = 1048576 + 1
        base_wav = generate_wav_bytes(duration_sec=1.0)
        padding = b"\x00" * (target_size - len(base_wav))
        over_bytes = base_wav + padding

        result = validator.validate(over_bytes, filename="over.wav")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.error_code, AudioErrorCode.FILE_TOO_LARGE.value)

    def test_audio_exactly_at_duration_limit(self):
        custom_settings = Settings(MAX_AUDIO_DURATION_SECONDS=10.0)
        validator = AudioValidator(config=custom_settings)
        exact_wav = generate_wav_bytes(duration_sec=10.0, sample_rate=16000)

        result = validator.validate(exact_wav, filename="exact_dur.wav")
        self.assertTrue(result.is_valid)
        self.assertTrue(math.isclose(result.duration_seconds, 10.0, abs_tol=0.05))

    def test_audio_just_above_duration_limit(self):
        custom_settings = Settings(MAX_AUDIO_DURATION_SECONDS=10.0)
        validator = AudioValidator(config=custom_settings)
        over_wav = generate_wav_bytes(duration_sec=10.2, sample_rate=16000)

        result = validator.validate(over_wav, filename="over_dur.wav")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.error_code, AudioErrorCode.AUDIO_DURATION_EXCEEDED.value)


class TestAudioValidatorSecurity(unittest.TestCase):
    """Security and path traversal protection tests."""

    def test_path_traversal_filename_sanitization(self):
        validator = AudioValidator()
        wav_data = generate_wav_bytes(duration_sec=1.0)
        result = validator.validate(wav_data, filename="../../../etc/passwd.wav")

        self.assertTrue(result.is_valid)
        self.assertEqual(result.format, "wav")

    def test_no_filesystem_path_leak_in_error_message(self):
        validator = AudioValidator()
        result = validator.validate(b"random_corrupted_data", filename="test.wav")

        self.assertFalse(result.is_valid)
        self.assertNotIn("c:\\", result.error_message.lower())
        assert "users" not in result.error_message.lower()
        self.assertNotIn("/", result.error_message)


if __name__ == "__main__":
    unittest.main()
