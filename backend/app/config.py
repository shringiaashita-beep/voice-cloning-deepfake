"""VoxGuard Configuration Module.

Provides typed configuration management for the backend using Pydantic Settings
with env variable override support and safe defaults.
"""

import os
from typing import Set

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field
except ImportError:
    try:
        from pydantic import BaseSettings, Field  # type: ignore
        SettingsConfigDict = None  # type: ignore
    except ImportError:
        # Standalone lightweight fallback if Pydantic is not yet installed in runtime
        class BaseSettings:  # type: ignore
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        def Field(default=None, **kwargs):  # type: ignore
            return default
        SettingsConfigDict = None  # type: ignore


class Settings(BaseSettings):
    """Application configuration settings for VoxGuard backend."""

    # File upload constraints
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=50,
        description="Maximum allowed audio upload file size in megabytes."
    )
    
    # Audio target processing standards
    TARGET_SAMPLE_RATE: int = Field(
        default=16000,
        description="Target sample rate in Hz for processing and feature extraction."
    )
    TARGET_CHANNELS: int = Field(
        default=1,
        description="Target channel count (1 for mono)."
    )

    # Allowed audio file formats
    ALLOWED_EXTENSIONS: Set[str] = Field(
        default={
            ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
            ".webm", ".opus", ".wma", ".aiff", ".aif", ".aifc",
            ".caf", ".amr", ".3gp", ".3gpp", ".mp4", ".oga"
        },
        description="Set of permitted lower-case audio file extensions."
    )
    ALLOWED_MIME_TYPES: Set[str] = Field(
        default={
            "audio/wav",
            "audio/x-wav",
            "audio/wave",
            "audio/mpeg",
            "audio/mp3",
            "audio/flac",
            "audio/x-flac",
            "audio/ogg",
            "audio/x-ogg",
            "application/ogg",
            "audio/mp4",
            "video/mp4",
            "audio/m4a",
            "audio/x-m4a",
            "audio/x-mp4",
            "audio/aac",
            "audio/x-aac",
            "audio/webm",
            "video/webm",
            "audio/opus",
            "audio/x-ms-wma",
            "audio/wma",
            "audio/aiff",
            "audio/x-aiff",
            "audio/x-caf",
            "audio/caf",
            "audio/amr",
            "audio/amr-wb",
            "audio/3gpp",
            "audio/3gpp2",
            "video/3gpp",
            "audio/basic",
            "audio/alac"
        },
        description="Set of permitted audio MIME types."
    )

    # Audio duration limits
    # Default: 86400.0 seconds (24 hours). Removes artificial duration limitations
    # while supporting any length from ultra-short voice snippets to hours of audio.
    MAX_AUDIO_DURATION_SECONDS: float = Field(
        default=86400.0,
        description="Maximum allowed audio duration in seconds (Default: 86400s / 24 hours)."
    )

    # Environment & CORS
    ENVIRONMENT: str = Field(
        default="development",
        description="Application environment (development, staging, production)."
    )
    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated list of allowed CORS origins."
    )

    # ML Classifier Settings
    MODEL_ENABLED: bool = Field(
        default=True,
        description="Master switch to activate real ML deepfake classifier adapter."
    )
    MODEL_PATH: str = Field(
        default="models/best_model.pt",
        description="Absolute or relative file path to pretrained model weights file."
    )
    MODEL_NAME: str = Field(
        default="VoxGuard Neural Deepfake Classifier",
        description="Human readable identifier for the active ML model."
    )
    MODEL_VERSION: str = Field(
        default="1.0.0",
        description="Semantic version string of the ML model."
    )
    MODEL_TYPE: str = Field(
        default="pytorch",
        description="Underlying ML framework architecture (e.g., pytorch, torchscript, onnx)."
    )

    @property
    def max_upload_size_bytes(self) -> int:
        """Returns the maximum upload file size in bytes."""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=True,
            extra="ignore"
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = True


# Instantiate global settings object
settings = Settings()
