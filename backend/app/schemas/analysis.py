"""Pydantic schemas for VoxGuard REST API requests and responses."""

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:  # type: ignore
        def dict(self, **kwargs):
            return self.__dict__
    def Field(default=None, **kwargs):  # type: ignore
        return default


class ErrorDetails(BaseModel):
    """Structured error message details."""
    code: str
    message: str


class ErrorResponse(BaseModel):
    """Structured API error response schema."""
    success: bool = False
    error: ErrorDetails


class HealthResponse(BaseModel):
    """API Health Check response schema."""
    status: str = "ok"
    service: str = "voxguard-api"
    version: str = "0.1.0"


class AnalysisResponse(BaseModel):
    """Complete analysis response schema returned by POST /api/v1/analyze."""
    success: bool = True
    audio_metadata: Dict[str, Any]
    classification: Dict[str, Any]
    visualization: Dict[str, Any]
    forensic_report: Dict[str, Any]
