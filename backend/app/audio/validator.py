"""Audio Validator Module for VoxGuard.

Performs robust multi-stage audio validation including:
1. Input existence & size checks.
2. Extension & MIME type verification.
3. Container magic number inspection.
4. Non-executing audio stream decoding & metadata extraction.
5. Duration, sample rate, channel, and non-emptiness validation.
"""

from dataclasses import dataclass, asdict
from enum import Enum
import io
import math
import os
from pathlib import Path
import struct
from typing import BinaryIO, Dict, Optional, Tuple, Union

from app.config import Settings, settings as global_settings


class AudioErrorCode(str, Enum):
    """Machine-readable error codes for audio validation failures."""
    MISSING_INPUT = "MISSING_INPUT"
    EMPTY_FILE = "EMPTY_FILE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_EXTENSION = "UNSUPPORTED_EXTENSION"
    UNSUPPORTED_MIME_TYPE = "UNSUPPORTED_MIME_TYPE"
    CORRUPTED_AUDIO = "CORRUPTED_AUDIO"
    AUDIO_DURATION_EXCEEDED = "AUDIO_DURATION_EXCEEDED"
    DECODING_FAILED = "DECODING_FAILED"


@dataclass
class AudioValidationResult:
    """Structured result returned by the audio validation pipeline."""
    is_valid: bool
    format: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    size_bytes: int = 0
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Converts result to a clean dictionary representation for API responses."""
        return asdict(self)


class AudioValidator:
    """Reusable audio validation service."""

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or global_settings

    def validate(
        self,
        file_input: Union[bytes, BinaryIO, None],
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> AudioValidationResult:
        """Runs the complete audio validation pipeline.

        Args:
            file_input: Raw audio file bytes or binary stream buffer.
            filename: Original file name provided during upload (sanitized internally).
            content_type: Optional MIME content-type string.

        Returns:
            AudioValidationResult object containing validation state and metadata.
        """
        # Step 1: Check input existence
        if file_input is None:
            return AudioValidationResult(
                is_valid=False,
                error_code=AudioErrorCode.MISSING_INPUT.value,
                error_message="No audio file or stream provided."
            )

        # Read input bytes cleanly without permanent filesystem storage
        if isinstance(file_input, bytes):
            file_bytes = file_input
        else:
            file_bytes = file_input.read()

        size_bytes = len(file_bytes)

        # Step 2: Check for empty input
        if size_bytes == 0:
            return AudioValidationResult(
                is_valid=False,
                size_bytes=0,
                error_code=AudioErrorCode.EMPTY_FILE.value,
                error_message="Uploaded audio file is empty (0 bytes)."
            )

        # Step 3: Check file size limit
        max_bytes = self.config.max_upload_size_bytes
        if size_bytes > max_bytes:
            max_mb = self.config.MAX_UPLOAD_SIZE_MB
            actual_mb = size_bytes / (1024 * 1024)
            return AudioValidationResult(
                is_valid=False,
                size_bytes=size_bytes,
                error_code=AudioErrorCode.FILE_TOO_LARGE.value,
                error_message=f"File size ({actual_mb:.2f} MB) exceeds maximum limit of {max_mb} MB."
            )

        # Step 4: Validate extension (preventing path traversal by stripping folder paths)
        ext = ""
        if filename:
            safe_filename = Path(filename).name
            ext = Path(safe_filename).suffix.lower()

        if ext and ext not in self.config.ALLOWED_EXTENSIONS:
            allowed_exts_str = ", ".join(sorted(self.config.ALLOWED_EXTENSIONS))
            return AudioValidationResult(
                is_valid=False,
                size_bytes=size_bytes,
                error_code=AudioErrorCode.UNSUPPORTED_EXTENSION.value,
                error_message=f"File extension '{ext}' is not supported. Allowed extensions: {allowed_exts_str}."
            )

        # Step 5: Validate MIME type if provided
        if content_type:
            clean_mime = content_type.split(";")[0].strip().lower()
            if clean_mime and clean_mime not in self.config.ALLOWED_MIME_TYPES:
                allowed_mimes_str = ", ".join(sorted(self.config.ALLOWED_MIME_TYPES))
                return AudioValidationResult(
                    is_valid=False,
                    size_bytes=size_bytes,
                    error_code=AudioErrorCode.UNSUPPORTED_MIME_TYPE.value,
                    error_message=f"Content-type '{clean_mime}' is not supported. Allowed MIME types: {allowed_mimes_str}."
                )

        # Step 6: Verify magic number container header
        detected_format = self._inspect_magic_header(file_bytes)
        if detected_format is None:
            return AudioValidationResult(
                is_valid=False,
                size_bytes=size_bytes,
                error_code=AudioErrorCode.CORRUPTED_AUDIO.value,
                error_message="Failed to recognize audio file header. File may be corrupted or not a valid audio format."
            )

        # Check container format against extension if extension was provided
        if ext and not self._format_matches_extension(detected_format, ext):
            return AudioValidationResult(
                is_valid=False,
                size_bytes=size_bytes,
                error_code=AudioErrorCode.CORRUPTED_AUDIO.value,
                error_message=f"File content format ({detected_format}) does not match file extension ({ext})."
            )

        # Step 7: Decode stream & extract audio properties
        try:
            duration_sec, sample_rate, channels = self._decode_audio_metadata(file_bytes, detected_format)
        except Exception as exc:
            return AudioValidationResult(
                is_valid=False,
                format=detected_format,
                size_bytes=size_bytes,
                error_code=AudioErrorCode.DECODING_FAILED.value,
                error_message="Audio stream decoding failed. File may be corrupted or contain invalid frames."
            )

        # Step 8: Validate non-empty decoded stream (accept any positive duration)
        if duration_sec <= 0.0001 or sample_rate <= 0 or channels <= 0:
            return AudioValidationResult(
                is_valid=False,
                format=detected_format,
                size_bytes=size_bytes,
                error_code=AudioErrorCode.EMPTY_FILE.value,
                error_message="Decoded audio stream contains zero frames or unreadable signal."
            )

        # Step 9: Check duration limit
        max_duration = self.config.MAX_AUDIO_DURATION_SECONDS
        if duration_sec > max_duration:
            return AudioValidationResult(
                is_valid=False,
                format=detected_format,
                duration_seconds=round(duration_sec, 3),
                sample_rate=sample_rate,
                channels=channels,
                size_bytes=size_bytes,
                error_code=AudioErrorCode.AUDIO_DURATION_EXCEEDED.value,
                error_message=f"Audio duration ({duration_sec:.2f}s) exceeds maximum allowed limit of {max_duration:.1f}s."
            )

        # Validation Successful!
        return AudioValidationResult(
            is_valid=True,
            format=detected_format,
            duration_seconds=round(duration_sec, 3),
            sample_rate=sample_rate,
            channels=channels,
            size_bytes=size_bytes
        )

    def _inspect_magic_header(self, data: bytes) -> Optional[str]:
        """Inspects byte stream magic header to identify format."""
        if len(data) < 4:
            return None

        # WAV: RIFF....WAVE
        if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WAVE":
            return "wav"

        # FLAC: fLaC
        if data.startswith(b"fLaC"):
            return "flac"

        # OGG / Opus: OggS
        if data.startswith(b"OggS"):
            return "ogg"

        # AAC (ADTS frame sync 12 bits 0xFFF, layer bits == 00)
        if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xF6) == 0xF0:
            return "aac"

        # MP3: ID3 tag OR MPEG audio frame sync (layer bits != 00)
        if data.startswith(b"ID3"):
            return "mp3"
        if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
            layer_bits = (data[1] >> 1) & 0x03
            if layer_bits != 0:
                return "mp3"

        # M4A / MP4: ....ftyp (ISO base media container)
        if len(data) >= 8 and data[4:8] == b"ftyp":
            return "m4a"

        # WebM / Matroska: \x1a\x45\xdf\xa3
        if data.startswith(b"\x1a\x45\xdf\xa3"):
            return "webm"

        # AIFF / AIFC: FORM....AIFF or FORM....AIFC
        if data.startswith(b"FORM") and len(data) >= 12 and (data[8:12] in (b"AIFF", b"AIFC")):
            return "aiff"

        # CAF: caff
        if data.startswith(b"caff"):
            return "caf"

        # AMR: #!AMR
        if data.startswith(b"#!AMR"):
            return "amr"

        # WMA / ASF: 0x30 0x26 0xb2 0x75
        if data.startswith(b"\x30\x26\xb2\x75"):
            return "wma"

        # AU / SND: .snd
        if data.startswith(b".snd"):
            return "au"

        # Dynamic fallback inspection via SoundFile
        try:
            import soundfile as sf
            with sf.SoundFile(io.BytesIO(data)) as sf_file:
                fmt = sf_file.format.lower()
                if fmt:
                    return fmt
        except Exception:
            pass

        return None

    def _format_matches_extension(self, detected_format: str, ext: str) -> bool:
        """Verifies container format matches file extension."""
        ext_clean = ext.lstrip(".").lower()
        format_clean = detected_format.lower()
        format_ext_map = {
            "wav": {"wav"},
            "flac": {"flac"},
            "ogg": {"ogg", "opus", "oga"},
            "opus": {"opus", "ogg"},
            "mp3": {"mp3"},
            "m4a": {"m4a", "mp4", "aac", "alac"},
            "mp4": {"mp4", "m4a", "aac"},
            "aac": {"aac", "m4a", "mp4"},
            "webm": {"webm", "mkv"},
            "aiff": {"aiff", "aif", "aifc"},
            "caf": {"caf"},
            "amr": {"amr", "3gp", "3gpp"},
            "wma": {"wma", "asf"},
            "au": {"au", "snd"},
        }
        if format_clean in format_ext_map:
            return ext_clean in format_ext_map[format_clean]
        return format_clean == ext_clean

    def _decode_audio_metadata(self, data: bytes, format_type: str) -> Tuple[float, int, int]:
        """Decodes metadata from audio bytes without loading full arrays into memory."""
        # 1. Try PySoundFile if installed
        try:
            import soundfile as sf
            with sf.SoundFile(io.BytesIO(data)) as sf_file:
                samplerate = sf_file.samplerate
                channels = sf_file.channels
                frames = len(sf_file)
                if 0 < frames < 100_000_000:
                    duration = frames / float(samplerate)
                    return float(duration), int(samplerate), int(channels)
        except Exception:
            pass

        # 2. Try Wave module for WAV files
        if format_type == "wav":
            import wave
            try:
                with wave.open(io.BytesIO(data), "rb") as wav_file:
                    channels = wav_file.getnchannels()
                    framerate = wav_file.getframerate()
                    nframes = wav_file.getnframes()
                    if framerate <= 0:
                        raise ValueError("Invalid sample rate")
                    duration = nframes / float(framerate)
                    return float(duration), int(framerate), int(channels)
            except Exception as e:
                raise ValueError(f"WAV decode error: {e}")

        # 3. Native parser for FLAC files STREAMINFO
        if format_type == "flac":
            try:
                if not data.startswith(b"fLaC"):
                    raise ValueError("Not a valid FLAC file")
                # FLAC STREAMINFO metadata block starts at byte 4
                block_header = data[4:8]
                block_type = block_header[0] & 0x7F
                if block_type != 0:
                    raise ValueError("First FLAC block is not STREAMINFO")
                
                streaminfo = data[8:42]
                if len(streaminfo) < 34:
                    raise ValueError("Truncated FLAC STREAMINFO header")

                sr_chan_bits_samples = int.from_bytes(streaminfo[10:18], byteorder="big")
                sample_rate = (sr_chan_bits_samples >> 44) & 0xFFFFF
                channels = ((sr_chan_bits_samples >> 41) & 0x07) + 1
                total_samples = sr_chan_bits_samples & 0xFFFFFFFFF

                if sample_rate <= 0:
                    raise ValueError("Invalid sample rate in FLAC header")

                if total_samples > 0:
                    duration = total_samples / float(sample_rate)
                    return float(duration), int(sample_rate), int(channels)
            except Exception:
                pass

        # 4. Native parser for OGG (Vorbis / Opus)
        if format_type == "ogg":
            try:
                # Basic OGG packet header parser
                if not data.startswith(b"OggS"):
                    raise ValueError("Not a valid OGG stream")
                
                # Look for OpusHead or \x01vorbis in first 200 bytes
                header_snippet = data[:200]
                if b"OpusHead" in header_snippet:
                    idx = header_snippet.index(b"OpusHead")
                    channels = header_snippet[idx + 9]
                    sample_rate = int.from_bytes(header_snippet[idx + 12:idx + 16], byteorder="little")
                    if sample_rate == 0:
                        sample_rate = 48000
                elif b"\x01vorbis" in header_snippet:
                    idx = header_snippet.index(b"\x01vorbis")
                    channels = header_snippet[idx + 11]
                    sample_rate = int.from_bytes(header_snippet[idx + 12:idx + 16], byteorder="little")
                else:
                    channels = 1
                    sample_rate = 16000

                # Duration estimation from size & bitrate or header page count
                duration = len(data) / (sample_rate * channels * 2.0)
                if duration <= 0:
                    duration = 1.0
                return float(duration), int(sample_rate), int(channels)
            except Exception as e:
                raise ValueError(f"OGG decode error: {e}")

        # 5. Native parser for MP3
        if format_type == "mp3":
            try:
                # Find first frame sync header (0xFFE0 / 0xFFF0)
                offset = 0
                if data.startswith(b"ID3"):
                    # Skip ID3v2 tag (10 bytes header + syncsafe size)
                    if len(data) >= 10:
                        tag_size = (
                            ((data[6] & 0x7F) << 21) |
                            ((data[7] & 0x7F) << 14) |
                            ((data[8] & 0x7F) << 7) |
                            (data[9] & 0x7F)
                        )
                        offset = 10 + tag_size

                # Search for frame header
                sample_rates_map = {
                    0: [44100, 48000, 32000], # MPEG 1
                    2: [22050, 24000, 16000], # MPEG 2
                    3: [11025, 12000, 8000]   # MPEG 2.5
                }
                
                sample_rate = 44100
                channels = 2
                found_header = False

                for i in range(offset, min(len(data) - 4, offset + 4096)):
                    if data[i] == 0xFF and (data[i+1] & 0xE0) == 0xE0:
                        header_int = int.from_bytes(data[i:i+4], byteorder="big")
                        version_bits = (header_int >> 19) & 0x03
                        sr_bits = (header_int >> 10) & 0x03
                        chan_bits = (header_int >> 6) & 0x03
                        
                        version_idx = 0 if version_bits == 3 else (2 if version_bits == 2 else 3)
                        if sr_bits != 3 and version_idx in sample_rates_map:
                            sample_rate = sample_rates_map[version_idx][sr_bits]
                            channels = 1 if chan_bits == 3 else 2
                            found_header = True
                            break

                if not found_header and not data.startswith(b"ID3"):
                    raise ValueError("No valid MP3 frame header found")

                # Estimate duration based on size
                audio_data_len = len(data) - offset
                duration = audio_data_len / (sample_rate * channels * 0.5) # approximate
                if duration <= 0:
                    duration = 1.0

                return float(duration), int(sample_rate), int(channels)
            except Exception as e:
                pass

        # 6. Universal decoder via FFmpeg (for M4A, AAC, WebM, WMA, AMR, Opus, AIFF, CAF, etc.)
        try:
            import imageio_ffmpeg
            import subprocess
            import soundfile as sf
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            proc = subprocess.run(
                [ffmpeg_exe, "-loglevel", "error", "-i", "pipe:0", "-vn", "-f", "wav", "pipe:1"],
                input=data,
                capture_output=True,
                timeout=12.0
            )
            if proc.returncode == 0 and len(proc.stdout) > 44:
                with sf.SoundFile(io.BytesIO(proc.stdout)) as sf_file:
                    samplerate = int(sf_file.samplerate)
                    channels = int(sf_file.channels)
                    frames = len(sf_file)
                    duration = frames / float(samplerate)
                    return float(duration), samplerate, channels
        except Exception:
            pass

        raise ValueError(f"Unsupported decoding format: {format_type}")
