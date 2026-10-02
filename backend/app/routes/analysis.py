"""FastAPI Analysis Router for VoxGuard.

Implements POST /api/v1/analyze orchestrating the complete end-to-end audio analysis pipeline:
Upload -> Validation -> Decoding -> Preprocessing -> Feature Extraction -> Baseline Classifier ->
Visualization Payload Builder -> Forensic Report Builder.
Enforces analysis-only status, null probabilities, structured error responses, and HTTP status mappings.
"""

import logging
import csv
import os
from typing import Any, Dict, Optional

try:
    from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
    from fastapi.responses import JSONResponse
except ImportError:
    # Stub placeholders if FastAPI is not installed in standard runtime
    APIRouter = Any  # type: ignore
    UploadFile = Any  # type: ignore
    JSONResponse = Any  # type: ignore
    Query = lambda default=None, **kwargs: default

from app.audio.decoder import AudioDecoder, AudioDecodingError
from app.audio.preprocessor import AudioPreprocessor, AudioPreprocessingError
from app.audio.validator import AudioErrorCode, AudioValidator
from app.audio.url_ingestor import AudioUrlIngestor, AudioUrlIngestionError
from app.config import settings
from app.features.extractor import AudioFeatureExtractor, FeatureExtractionError
from app.features.visualization import VisualizationBuilder
from app.forensics.report_builder import ForensicReportBuilder
from app.forensics.speaker_comparator import SpeakerComparator
from app.models.registry import model_registry
from app.schemas.analysis import AnalysisResponse, ErrorDetails, ErrorResponse
from app.evaluation.manifest_validator import ManifestRow
from app.evaluation.speaker_split_validator import SpeakerSplitValidator
from app.training.dataset import DatasetLoaderError, FeatureDataset


logger = logging.getLogger("voxguard.analysis")

try:
    router = APIRouter(prefix="/api/v1", tags=["Analysis"])
except Exception:
    router = None


def map_error_code_to_status(error_code: str) -> int:
    """Maps internal AudioErrorCode strings to appropriate HTTP status codes."""
    if error_code in (AudioErrorCode.MISSING_INPUT.value, AudioErrorCode.EMPTY_FILE.value):
        return 400  # Bad Request
    if error_code == AudioErrorCode.FILE_TOO_LARGE.value:
        return 413  # Payload Too Large
    if error_code in (AudioErrorCode.UNSUPPORTED_EXTENSION.value, AudioErrorCode.UNSUPPORTED_MIME_TYPE.value):
        return 415  # Unsupported Media Type
    if error_code in (AudioErrorCode.CORRUPTED_AUDIO.value, AudioErrorCode.AUDIO_DURATION_EXCEEDED.value, AudioErrorCode.DECODING_FAILED.value):
        return 422  # Unprocessable Entity
    return 400


def analyze_audio_stream(
    file_bytes: bytes,
    filename: str = "upload.wav",
    content_type: str = "audio/wav",
    model_id: Optional[str] = None
) -> Dict[str, Any]:
    """Pure pipeline orchestrator function used by FastAPI endpoint and test runners."""
    # Step 1: Validation
    validator = AudioValidator(config=settings)
    val_result = validator.validate(file_bytes, filename=filename, content_type=content_type)
    if not val_result.is_valid:
        http_code = map_error_code_to_status(val_result.error_code or "")
        return {
            "http_status": http_code,
            "response": {
                "success": False,
                "error": {
                    "code": val_result.error_code,
                    "message": val_result.error_message
                }
            }
        }

    # Step 2: Decoding
    try:
        decoder = AudioDecoder(validator=validator)
        decoded = decoder.decode(file_bytes, filename=filename, content_type=content_type)
    except AudioDecodingError as exc:
        return {
            "http_status": 422,
            "response": {
                "success": False,
                "error": {
                    "code": AudioErrorCode.DECODING_FAILED.value,
                    "message": "Audio stream decoding failed."
                }
            }
        }

    # Step 3: Preprocessing
    try:
        preprocessor = AudioPreprocessor(config=settings)
        preprocessed = preprocessor.process(decoded)
    except AudioPreprocessingError as exc:
        return {
            "http_status": 422,
            "response": {
                "success": False,
                "error": {
                    "code": "PREPROCESSING_FAILED",
                    "message": "Audio preprocessing or resampling failed."
                }
            }
        }

    # Step 4: Feature Extraction
    try:
        extractor = AudioFeatureExtractor()
        features = extractor.extract_all(preprocessed)
    except FeatureExtractionError as exc:
        return {
            "http_status": 422,
            "response": {
                "success": False,
                "error": {
                    "code": "FEATURE_EXTRACTION_FAILED",
                    "message": "Feature extraction failed."
                }
            }
        }

    # Step 5: Model Inference (from ModelRegistry)
    if model_id and model_id in model_registry._models:
        classifier = model_registry.get_model(model_id)
    else:
        classifier = model_registry.get_active_model()
    classification = classifier.predict(features)

    # Step 6: Visualization Payload Building
    viz_builder = VisualizationBuilder()
    visualization = viz_builder.build_all(preprocessed, features.log_mel_spectrogram)

    # Step 7: Forensic Report Generation
    report_builder = ForensicReportBuilder()
    forensic_report = report_builder.build_report(classification, features, preprocessed)

    # Step 8: Return complete structured analysis dictionary
    return {
        "http_status": 200,
        "response": {
            "success": True,
            "audio_metadata": preprocessed.to_dict(),
            "classification": classification.to_dict(),
            "visualization": visualization.to_dict(),
            "forensic_report": forensic_report.to_dict()
        }
    }


if router is not None:
    @router.post("/analyze", response_model=AnalysisResponse)
    async def analyze_audio(
        file: UploadFile = File(...),
        model: Optional[str] = Query(None)
    ):
        """FastAPI route handler for POST /api/v1/analyze."""
        if not file:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": {"code": "MISSING_INPUT", "message": "No audio file provided in request."}}
            )

        try:
            file_bytes = await file.read()
            filename = file.filename or "upload.wav"
            content_type = file.content_type or "audio/wav"

            result = analyze_audio_stream(file_bytes, filename=filename, content_type=content_type, model_id=model)
            return JSONResponse(
                status_code=result["http_status"],
                content=result["response"]
            )
        except Exception as exc:
            logger.error("Unexpected error in /analyze endpoint: %s", exc, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "An unexpected server error occurred during audio analysis."
                    }
                }
            )

    @router.post("/analyze/url")
    async def analyze_audio_url(payload: Dict[str, Any]):
        """FastAPI route handler for POST /api/v1/analyze/url.
        Safely fetches remote audio via HTTP/HTTPS and runs forensic pipeline.
        """
        url = payload.get("url")
        model = payload.get("model")
        if not url:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": {"code": "MISSING_URL", "message": "No audio URL provided in request payload."}}
            )

        ingestor = AudioUrlIngestor()
        try:
            file_bytes, filename, content_type = ingestor.fetch(url)
        except AudioUrlIngestionError as exc:
            return JSONResponse(
                status_code=422,
                content={"success": False, "error": {"code": "URL_INGESTION_FAILED", "message": str(exc)}}
            )
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": {"code": "URL_FETCH_ERROR", "message": f"Unexpected error fetching URL: {str(exc)}"}}
            )

        result = analyze_audio_stream(file_bytes, filename=filename, content_type=content_type, model_id=model)
        return JSONResponse(
            status_code=result["http_status"],
            content=result["response"]
        )

    @router.post("/compare")
    async def compare_speakers(
        sample_a: UploadFile = File(...),
        sample_b: UploadFile = File(...),
        model: Optional[str] = Query(None)
    ):
        """FastAPI route handler for POST /api/v1/compare.
        Performs dual acoustic analysis and speaker biometric comparison.
        """
        if not sample_a or not sample_b:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": {"code": "MISSING_SAMPLES", "message": "Both sample_a and sample_b files are required."}}
            )

        bytes_a = await sample_a.read()
        bytes_b = await sample_b.read()
        fname_a = sample_a.filename or "questioned_sample.wav"
        fname_b = sample_b.filename or "reference_sample.wav"

        # 1. Analyze Sample A
        res_a = analyze_audio_stream(bytes_a, filename=fname_a, content_type=sample_a.content_type or "audio/wav", model_id=model)
        if res_a["http_status"] != 200:
            return JSONResponse(status_code=res_a["http_status"], content=res_a["response"])

        # 2. Analyze Sample B
        res_b = analyze_audio_stream(bytes_b, filename=fname_b, content_type=sample_b.content_type or "audio/wav", model_id=model)
        if res_b["http_status"] != 200:
            return JSONResponse(status_code=res_b["http_status"], content=res_b["response"])

        # 3. Extract features and perform biometric comparison
        validator = AudioValidator()
        decoder = AudioDecoder(validator=validator)
        preprocessor = AudioPreprocessor()
        extractor = AudioFeatureExtractor()

        prep_a = preprocessor.process(decoder.decode(bytes_a, filename=fname_a))
        prep_b = preprocessor.process(decoder.decode(bytes_b, filename=fname_b))

        feat_a = extractor.extract_all(prep_a)
        feat_b = extractor.extract_all(prep_b)

        classifier = model_registry.get_model(model) if (model and model in model_registry._models) else model_registry.get_active_model()
        cls_a = classifier.predict(feat_a)
        cls_b = classifier.predict(feat_b)

        comparator = SpeakerComparator()
        comp_result = comparator.compare(
            features_a=feat_a,
            classification_a=cls_a,
            features_b=feat_b,
            classification_b=cls_b,
            filename_a=fname_a,
            filename_b=fname_b
        )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "comparison": comp_result.to_dict(),
                "sample_a_analysis": res_a["response"],
                "sample_b_analysis": res_b["response"]
            }
        )

    @router.get("/models")
    async def list_models():
        """FastAPI route handler for GET /api/v1/models.

        Returns safe summary list of registered model metadata and operational statuses
        without disclosing internal filesystem paths.
        """
        raw_models = model_registry.list_models()
        model_list = [
            {
                "id": m_id,
                "name": info["name"],
                "version": info["version"],
                "status": info["status"],
                "is_active": info["is_active"]
            }
            for m_id, info in raw_models.items()
        ]
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "models": model_list
            }
        )

    @router.post("/models/select")
    async def select_model(payload: Dict[str, Any]):
        """FastAPI route handler for POST /api/v1/models/select.

        Switches the globally active classifier model in ModelRegistry.
        """
        model_id = payload.get("model_id")
        if not model_id or model_id not in model_registry._models:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": {"code": "INVALID_MODEL_ID", "message": f"Model '{model_id}' is not registered."}}
            )
        model_registry.set_active_model(model_id)
        return JSONResponse(
            status_code=200,
            content={"success": True, "active_model_id": model_id}
        )

    @router.get("/training/status")
    async def get_training_status():
        """FastAPI route handler for GET /api/v1/training/status.

        Returns training dataset infrastructure status and readiness metrics
        without returning fake probabilities or model confidence scores.
        """
        feature_dir = os.path.abspath("datasets/prepared/features")
        manifest_path = os.path.join(feature_dir, "features_manifest.csv")

        if not os.path.exists(manifest_path):
            manifest_path = os.path.abspath("data/features/features_manifest.csv")
            feature_dir = os.path.dirname(manifest_path)

        empty_status = {
            "dataset_available": False,
            "feature_artifacts": 0,
            "windows": 0,
            "speakers": 0,
            "class_balance": {"human": 0, "synthetic": 0},
            "speaker_leakage": False,
            "feature_integrity": False,
            "ood_available": False,
            "training_ready": False,
            "validation_errors": ["Feature manifest is unavailable."],
        }

        if not os.path.exists(manifest_path):
            return JSONResponse(status_code=200, content=empty_status)

        try:
            manifest_rows = []
            validation_errors = []

            with open(manifest_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                required_columns = {"feature_path", "label", "speaker_id", "source", "source_id", "split"}
                actual_columns = {column.strip().lower() for column in (reader.fieldnames or [])}
                missing_columns = sorted(required_columns - actual_columns)
                if missing_columns:
                    validation_errors.append(f"Feature manifest is missing columns: {', '.join(missing_columns)}.")

                for row in reader:
                    manifest_rows.append(row)

            if validation_errors:
                return JSONResponse(status_code=200, content={**empty_status, "validation_errors": validation_errors})

            typed_rows = [
                ManifestRow(
                    file_path=(row.get("feature_path") or "").strip(),
                    label=(row.get("label") or "").strip().lower(),
                    speaker_id=(row.get("speaker_id") or "").strip(),
                    source=(row.get("source") or "").strip(),
                    source_id=(row.get("source_id") or "").strip(),
                    split=(row.get("split") or "").strip().lower(),
                )
                for row in manifest_rows
            ]
            split_report = SpeakerSplitValidator().validate_speaker_splits(typed_rows)
            window_counts = {"human": 0, "synthetic": 0}
            total_windows = 0
            feature_integrity = True
            for split in ("train", "validation", "test", "ood"):
                try:
                    dataset = FeatureDataset(manifest_path, feature_dir=feature_dir, split=split)
                    total_windows += len(dataset.samples)
                    for sample in dataset.samples:
                        window_counts[sample.label_str] += 1
                except DatasetLoaderError as exc:
                    feature_integrity = False
                    validation_errors.append(f"{split} split: {str(exc)}")

            artifact_count = len(manifest_rows)
            has_ood = any(row.split == "ood" for row in typed_rows)
            training_ready = (
                artifact_count > 0
                and feature_integrity
                and split_report.is_speaker_disjoint
                and window_counts["human"] > 0
                and window_counts["synthetic"] > 0
            )

            return JSONResponse(
                status_code=200,
                content={
                    "dataset_available": True,
                    "feature_artifacts": artifact_count,
                    "windows": total_windows,
                    "speakers": split_report.total_speakers,
                    "class_balance": window_counts,
                    "speaker_leakage": not split_report.is_speaker_disjoint,
                    "feature_integrity": feature_integrity,
                    "ood_available": has_ood,
                    "training_ready": training_ready,
                    "validation_errors": validation_errors + split_report.errors,
                }
            )
        except Exception as exc:
            logger.error("Training readiness validation failed: %s", exc, exc_info=True)
            return JSONResponse(status_code=200, content={**empty_status, "validation_errors": ["Training readiness validation failed."]})

