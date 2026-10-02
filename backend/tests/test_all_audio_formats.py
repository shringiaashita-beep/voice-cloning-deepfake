"""Unit and Integration Tests for Universal Multi-Format Audio Support in VoxGuard.

Verifies that all supported audio formats (WAV, MP3, FLAC, OGG, M4A, AAC, WEBM, AIFF, WMA):
1. Pass container validation and magic header inspection.
2. Successfully decode into uncompressed float32 PCM arrays via AudioDecoder.
3. Pass through preprocessing and feature extraction without error.
4. Return 200 OK with complete forensic and visualization payloads in analyze_audio_stream.
5. Correctly reject unsupported formats and malicious extension mismatches.
"""

import io
import math
import subprocess
import unittest
import wave

import numpy as np

from app.audio.decoder import AudioDecoder
from app.audio.validator import AudioErrorCode, AudioValidator
from app.config import settings
from app.routes.analysis import analyze_audio_stream


def generate_synthesized_wav(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generates standard PCM 16-bit WAV bytes in memory."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    pcm = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def transcode_audio(input_wav: bytes, ffmpeg_args: list) -> bytes:
    """Transcodes WAV bytes to specified target format via imageio-ffmpeg."""
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe, "-loglevel", "error", "-i", "pipe:0"] + ffmpeg_args + ["pipe:1"]
    proc = subprocess.run(cmd, input=input_wav, capture_output=True, timeout=10.0)
    if proc.returncode != 0 or len(proc.stdout) == 0:
        raise RuntimeError(f"FFmpeg transcode failed: {proc.stderr.decode('utf-8', errors='ignore')}")
    return proc.stdout


class TestAllAudioFormats(unittest.TestCase):
    """Verifies complete multi-format audio handling across VoxGuard."""

    @classmethod
    def setUpClass(cls):
        cls.base_wav = generate_synthesized_wav(duration_sec=1.5, sample_rate=16000)
        cls.validator = AudioValidator(config=settings)
        cls.decoder = AudioDecoder(validator=cls.validator)

        # Generate audio payloads in memory
        cls.formats = {
            "wav": {
                "ext": ".wav",
                "mime": "audio/wav",
                "bytes": cls.base_wav,
            },
            "mp3": {
                "ext": ".mp3",
                "mime": "audio/mpeg",
                "bytes": transcode_audio(cls.base_wav, ["-f", "mp3"]),
            },
            "flac": {
                "ext": ".flac",
                "mime": "audio/flac",
                "bytes": transcode_audio(cls.base_wav, ["-f", "flac"]),
            },
            "ogg": {
                "ext": ".ogg",
                "mime": "audio/ogg",
                "bytes": transcode_audio(cls.base_wav, ["-f", "ogg", "-c:a", "libvorbis"]),
            },
            "webm": {
                "ext": ".webm",
                "mime": "audio/webm",
                "bytes": transcode_audio(cls.base_wav, ["-f", "webm", "-c:a", "libopus"]),
            },
            "m4a": {
                "ext": ".m4a",
                "mime": "audio/mp4",
                "bytes": transcode_audio(cls.base_wav, ["-f", "mp4", "-c:a", "aac", "-movflags", "frag_keyframe+empty_moov"]),
            },
            "aac": {
                "ext": ".aac",
                "mime": "audio/aac",
                "bytes": transcode_audio(cls.base_wav, ["-f", "adts", "-c:a", "aac"]),
            },
            "aiff": {
                "ext": ".aiff",
                "mime": "audio/aiff",
                "bytes": transcode_audio(cls.base_wav, ["-f", "aiff"]),
            },
            "wma": {
                "ext": ".wma",
                "mime": "audio/x-ms-wma",
                "bytes": transcode_audio(cls.base_wav, ["-f", "asf", "-c:a", "wmav2"]),
            },
            "mp4": {
                "ext": ".mp4",
                "mime": "video/mp4",
                "bytes": transcode_audio(cls.base_wav, ["-f", "mp4", "-c:a", "aac", "-movflags", "frag_keyframe+empty_moov"]),
            },
            "mp4_audio": {
                "ext": ".mp4",
                "mime": "audio/mp4",
                "bytes": transcode_audio(cls.base_wav, ["-f", "mp4", "-c:a", "aac", "-movflags", "frag_keyframe+empty_moov"]),
            },
        }

    def test_all_formats_pass_validation(self):
        """1. Verifies that all formats pass container validation."""
        for fmt_name, data in self.formats.items():
            filename = f"sample_{fmt_name}{data['ext']}"
            result = self.validator.validate(data["bytes"], filename=filename, content_type=data["mime"])
            self.assertTrue(
                result.is_valid,
                f"Validation failed for format {fmt_name}: {result.error_code} - {result.error_message}"
            )
            self.assertGreater(result.duration_seconds, 0.5)
            self.assertGreater(result.sample_rate, 0)
            self.assertGreater(result.channels, 0)

    def test_all_formats_decode_to_pcm(self):
        """2. Verifies that AudioDecoder successfully decodes all formats to float32 PCM."""
        for fmt_name, data in self.formats.items():
            filename = f"sample_{fmt_name}{data['ext']}"
            decoded = self.decoder.decode(data["bytes"], filename=filename, content_type=data["mime"])
            self.assertIsNotNone(decoded)
            self.assertIsInstance(decoded.pcm_data, np.ndarray)
            self.assertEqual(decoded.pcm_data.dtype, np.float32)
            self.assertGreater(len(decoded.pcm_data), 0)
            self.assertGreater(decoded.duration_seconds, 0.5)

    def test_all_formats_pass_end_to_end_analysis(self):
        """3. Verifies that POST /api/v1/analyze orchestrator successfully processes all formats."""
        for fmt_name, data in self.formats.items():
            filename = f"sample_{fmt_name}{data['ext']}"
            res = analyze_audio_stream(
                data["bytes"],
                filename=filename,
                content_type=data["mime"],
                model_id="voice_clone_detector"
            )
            self.assertEqual(
                res["http_status"], 200,
                f"End-to-end analysis failed for format {fmt_name}: {res}"
            )
            self.assertTrue(res["response"]["success"])
            self.assertIn("classification", res["response"])
            self.assertIn("visualization", res["response"])
            self.assertIn("forensic_report", res["response"])
            self.assertIn("audio_metadata", res["response"])
            self.assertGreater(len(res["response"]["visualization"]["waveform"]["peak_envelope"]), 0)

    def test_unsupported_extension_rejected(self):
        """4. Verifies non-audio extensions are rejected with HTTP 415 UNSUPPORTED_EXTENSION."""
        res = analyze_audio_stream(self.base_wav, filename="malicious.exe", content_type="application/octet-stream")
        self.assertEqual(res["http_status"], 415)
        self.assertEqual(res["response"]["error"]["code"], AudioErrorCode.UNSUPPORTED_EXTENSION.value)

    def test_disallowed_mime_rejected(self):
        """5. Verifies disallowed MIME types are rejected with HTTP 415 UNSUPPORTED_MIME_TYPE."""
        res = analyze_audio_stream(self.base_wav, filename="sample.wav", content_type="application/pdf")
        self.assertEqual(res["http_status"], 415)
        self.assertEqual(res["response"]["error"]["code"], AudioErrorCode.UNSUPPORTED_MIME_TYPE.value)

    def test_extension_content_mismatch_rejected(self):
        """6. Verifies extension spoofing is detected with CORRUPTED_AUDIO."""
        # Provide MP3 payload with .wav extension
        mp3_bytes = self.formats["mp3"]["bytes"]
        val = self.validator.validate(mp3_bytes, filename="spoofed.wav", content_type="audio/wav")
        self.assertFalse(val.is_valid)
        self.assertEqual(val.error_code, AudioErrorCode.CORRUPTED_AUDIO.value)


if __name__ == "__main__":
    unittest.main()
