"""VoxGuard FastAPI Application Entrypoint.

Provides API initialization, CORS middleware configuration, route registration,
health check endpoints, and structured exception handling.
"""

import logging
from typing import Any, Dict

try:
    from fastapi import FastAPI, Request, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:
    FastAPI = None  # type: ignore

from app.config import settings
from app.routes.analysis import router as analysis_router
from app.schemas.analysis import HealthResponse


logger = logging.getLogger("voxguard.main")


def create_app():
    """Application factory for FastAPI instance."""
    if FastAPI is None:
        return None

    app = FastAPI(
        title="VoxGuard API",
        description="AI Audio Deepfake Detection & Forensic Analysis Platform API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Configure CORS Middleware using settings
    allowed_origins = [
        origin.strip()
        for origin in settings.ALLOWED_ORIGINS.split(",")
        if origin.strip()
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register Routers
    if analysis_router:
        app.include_router(analysis_router)

    # Health check endpoints
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    @app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
    def health_check():
        """Returns service operational status."""
        return {
            "status": "ok",
            "service": "voxguard-api",
            "version": "0.1.0"
        }

    # Custom Exception Handlers for Structured Errors without stack trace disclosure
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": {
                    "code": "NOT_FOUND",
                    "message": "The requested resource or endpoint was not found."
                }
            }
        )

    @app.exception_handler(422)
    async def unprocessable_entity_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "code": "UNPROCESSABLE_ENTITY",
                    "message": "Validation error in request payload or parameters."
                }
            }
        )

    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc: Exception):
        logger.error("Unhandled internal exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected server error occurred."
                }
            }
        )

    return app


app = create_app()
