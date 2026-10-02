"""Unit tests for VoxGuard Audio Decoder Service.

Tests decoding of WAV, MP3, FLAC, and OGG formats into DecodedAudio PCM arrays,
as well as error handling for corrupted or invalid streams.
Inherits from unittest.TestCase for standalone and pytest compatibility.
"""

import io
import math
import struct
import unittest
import wave

import numpy as np

from app.audio.decoder import AudioDecoder, AudioDecodingError, DecodedAudio


# Helper functions to synthesize test audio payloads

def generate_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Generates valid PCM WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)
        num_frames = int(duration_sec * sample_rate)
        # Create a simple 440 Hz sine wave
        samples = []
        for i in range(num_frames):
            val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            data = struct.pack("<h", val)
            if channels == 2:
                data = data + data
            samples.append(data)
        wav_file.writeframesraw(b"".join(samples))
    return buf.getvalue()


def generate_flac_bytes(duration_sec: float = 1.0, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Generates valid FLAC header bytes."""
    header = b"fLaC"
    block_header = b"\x00\x00\x00\x22"
    blocks = b"\x10\x00\x10\x00" + b"\x00\x00\x00\x00\x00\x00"
    total_samples = int(duration_sec * sample_rate)
    sr_chan_bits_samples = (sample_rate << 44) | ((channels - 1) << 41) | (15 << 36) | (total_samples & 0xFFFFFFFFF)
    sr_chan_bytes = sr_chan_bits_samples.to_bytes(8, byteorder="big")
    md5_hash = b"\x00" * 16
    return header + block_header + blocks + sr_chan_bytes + md5_hash


def generate_mp3_bytes(duration_sec: float = 1.0) -> bytes:
    """Generates valid synthetic MP3 payload."""
    id3_tag = b"ID3\x04\x00\x00\x00\x00\x00\x0aTITLE\x00\x00\x00"
    frame_header = b"\xff\xfb\x90\x00"
    payload = b"\x00" * int(duration_sec * 44100 * 2 * 0.5)
    return id3_tag + frame_header + payload


def generate_ogg_bytes(duration_sec: float = 1.0) -> bytes:
    """Generates valid synthetic OGG payload."""
    header = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00"
    body = b"OpusHead\x01\x01\x38\x01\x80\xbb\x00\x00\x00\x00\x00"
    payload = b"\x00" * int(duration_sec * 16000 * 2)
    return header + body + payload


class TestAudioDecoder(unittest.TestCase):
    """Test suite for AudioDecoder service."""

    def setUp(self):
        self.decoder = AudioDecoder()

    def test_decode_valid_wav(self):
        wav_bytes = generate_wav_bytes(duration_sec=1.5, sample_rate=16000, channels=1)
        decoded: DecodedAudio = self.decoder.decode(wav_bytes, filename="test.wav")

        self.assertIsInstance(decoded.pcm_data, np.ndarray)
        self.assertEqual(decoded.format, "wav")
        self.assertEqual(decoded.sample_rate, 16000)
        self.assertEqual(decoded.channels, 1)
        self.assertTrue(math.isclose(decoded.duration_seconds, 1.5, abs_tol=0.05))
        self.assertGreater(len(decoded.pcm_data), 0)

    def test_decode_valid_mp3(self):
        mp3_bytes = generate_mp3_bytes(duration_sec=1.0)
        decoded: DecodedAudio = self.decoder.decode(mp3_bytes, filename="test.mp3")

        self.assertIsInstance(decoded.pcm_data, np.ndarray)
        self.assertEqual(decoded.format, "mp3")
        self.assertGreater(decoded.num_frames, 0)

    def test_decode_valid_flac(self):
        flac_bytes = generate_flac_bytes(duration_sec=2.0, sample_rate=44100, channels=2)
        decoded: DecodedAudio = self.decoder.decode(flac_bytes, filename="test.flac")

        self.assertIsInstance(decoded.pcm_data, np.ndarray)
        self.assertEqual(decoded.format, "flac")
        self.assertEqual(decoded.sample_rate, 44100)
        self.assertEqual(decoded.channels, 2)
        self.assertTrue(math.isclose(decoded.duration_seconds, 2.0, abs_tol=0.05))

    def test_decode_valid_ogg(self):
        ogg_bytes = generate_ogg_bytes(duration_sec=1.0)
        decoded: DecodedAudio = self.decoder.decode(ogg_bytes, filename="test.ogg")

        self.assertIsInstance(decoded.pcm_data, np.ndarray)
        self.assertEqual(decoded.format, "ogg")

    def test_decode_corrupted_audio(self):
        garbage_bytes = b"RIFF" + b"\x00" * 20
        with self.assertRaises(AudioDecodingError):
            self.decoder.decode(garbage_bytes, filename="corrupt.wav")

    def test_decode_unsupported_file_extension(self):
        text_bytes = b"This is plain text data"
        with self.assertRaises(AudioDecodingError):
            self.decoder.decode(text_bytes, filename="document.pdf")


if __name__ == "__main__":
    unittest.main()
