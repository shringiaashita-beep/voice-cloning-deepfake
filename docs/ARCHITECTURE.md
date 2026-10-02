# VoxGuard Architectural Specification

## Overview

VoxGuard follows a decoupled, multi-layered architecture designed for modularity, maintainability, and forensic explainability.

---

## 1. System Layers & Responsibilities

```
+-------------------------------------------------------------+
|                      Frontend Layer                         |
|  Next.js (App Router) + React + Tailwind CSS + TypeScript   |
+-------------------------------------------------------------+
                               |
                               | REST API (HTTP JSON / Multipart)
                               v
+-------------------------------------------------------------+
|                        API Layer                            |
|        FastAPI Routes & Request Validation Middleware       |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                 Audio Processing Layer                      |
|  File Validation -> Decoding -> Dual-Path Audio Routing     |
+-------------------------------------------------------------+
                 /                           \
                /                             \
               v                               v
+------------------------------+  +---------------------------+
|      Canonical ML Path       |  |   Forensic Analysis Path  |
|  16kHz Mono float32 PCM      |  |  Original Sample Rate &   |
|  Standard speech features    |  |  Full Frequency Spectrum  |
+------------------------------+  +---------------------------+
               |                               |
               v                               v
+------------------------------+  +---------------------------+
|    Feature Extraction Engine |  |  High-Frequency Anomaly   |
|  Log-Mel, MFCC, Centroid,    |  |  & Phase Artifact Detector|
|  Rolloff (NumPy Arrays)      |  |  (Preserves > 8 kHz data) |
+------------------------------+  +---------------------------+
         /           \                         |
        /             \                        |
       v               v                       v
+-------------+  +--------------------------------------------+
| Model       |  |       Visualization Payload Builder        |
| Registry &  |  |   Downsampled ~300pt waveform envelope &   |
| Classifiers |  |   normalized 40x120 spectrogram grid       |
+-------------+  +--------------------------------------------+
       \                               /
        \                             /
         v                           v
+-------------------------------------------------------------+
|                 Forensic Reporting Layer                    |
|    Verdict Assembly, Provenance Tracking & Explanations     |
+-------------------------------------------------------------+
```
                               |
                               v
                        API Response Payload
```

---

## 1.1. ML Inference Architecture & Model Adapter Pattern

VoxGuard provides a pluggable **Machine Learning Classifier Architecture** through the `ModelRegistry` and `AbstractDeepfakeClassifier` interface:

```
                      +-----------------------------+
                      |       FastAPI Router        |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |        ModelRegistry        |
                      +-----------------------------+
                                /         \
                               /           \
                              v             v
       +--------------------------+     +--------------------------+
       |   BaselineClassifier     |     |   RealMLClassifier       |
       | status: "analysis_only"  |     |   Adapter (PyTorch/ONNX) |
       | probabilities: null      |     | status: "ready"/"not_ready"|
       +--------------------------+     +--------------------------+
```

### Key ML Architecture Rules:

1. **Framework Decoupling**: FastAPI routes interact solely with `ModelRegistry.get_active_model()`. The backend route code never calls PyTorch or ONNX directly.
2. **Explicit Weight Loading**: No model weights or datasets are downloaded automatically during import, startup, requests, or tests. Loading is explicitly triggered via `load_model()` from on-disk paths specified in `MODEL_PATH`.
3. **Configuration Control**: Environment variables control ML activation (`MODEL_ENABLED`, `MODEL_PATH`, `MODEL_NAME`, `MODEL_VERSION`, `MODEL_TYPE`).
4. **Graceful Fallback**: If `MODEL_ENABLED=false` or if `MODEL_PATH` points to a missing file, the adapter defaults to `status = "not_ready"` and VoxGuard operates cleanly on the `BaselineClassifier` (`status = "analysis_only"`), returning `probabilities: null`.
5. **Provenance Isolation**:
   - `metadata`: Stream format and sample rate.
   - `signal_analysis`: Objective DSP acoustic observations (centroid, rolloff, log-mel energy).
   - `heuristic`: System state indicators (`ANALYSIS ONLY` or `MODEL NOT READY`).
   - `ml_model`: Evaluated ML decision verdict and confidence scores (only active when `status == "ready"`).

---

## 1.2. Model Integration Readiness & Manifest System

> [!IMPORTANT]
> **VoxGuard Policy**: VoxGuard does not consider a model production-ready merely because inference code executes without throwing errors. Production-ready `status = "ready"` requires documented evaluation metrics, zero-speaker-overlap split verification, and valid manifest checksums.

### Model Integration Abstractions:

1. **ModelInputContract (`backend/app/models/model_input.py`)**:
   - Canonical 16,000 Hz Mono `float32` PCM array.
   - Standard 80-bin Log-Mel Spectrogram grid shape `[Batch, Channels, Freq, Time]`.
   - Peak magnitude normalization `[-1.0, 1.0]`.

2. **ModelOutputContract (`backend/app/models/model_output.py`)**:
   - Probabilities (`human`, `synthetic`) MUST be `None` unless status is `READY`.
   - When `READY`, $P(\text{human}) + P(\text{synthetic}) \approx 1.0$.
   - Confidence score derived directly from output logits/softmax.

3. **ModelManifest (`backend/app/models/model_manifest.py`)**:
   - Defines external checkpoint metadata, SHA-256 checksums, architecture, class labels, training/evaluation dataset provenance, and speaker-disjoint split policies.

4. **ModelValidator (`backend/app/models/model_validator.py`)**:
   - Performs passive validation of model format, file readability, checksums, and tensor shapes. **Never executes untrusted arbitrary binary code or downloads external assets.**

5. **Probability Calibration Spec (`backend/app/evaluation/calibration.py`)**:
   - Supports Temperature Scaling, Platt Scaling, Isotonic Regression. Reports `calibration_status = "not_calibrated"` until calibrated weights artifact exists.

6. **ModelLoader (`backend/app/models/model_loader.py`)**:
   - Performs offline discovery of local model weight checkpoints (`.pt`, `.onnx`) and JSON manifest descriptors (`models/manifest.json`). Automatically validates checksums and registers valid classifiers in `ModelRegistry` without external network connections.

---

## 2. End-to-End Processing Pipeline

```
HTTP Multipart Upload
   │
   ▼
[1. Input Security & Validation] (AudioValidator - size < 10MB, extension, MIME, magic header)
   │
   ▼
[2. Stream Decoding] (AudioDecoder - PySoundFile C-native backend)
   │  Returns: DecodedAudio (raw PCM array, orig_sr, orig_channels)
   │
   ├──► [Forensic Analysis Path] (Preserves original SR > 8kHz bandwidth)
   │
   ▼
[3. Canonical ML Processing] (AudioPreprocessor)
   ├─ Average channels to 1D mono
   ├─ Resample to 16,000 Hz (Polyphase FIR filter)
   ├─ Convert to np.float32
   └─ Peak normalize magnitude to [-1.0, 1.0]
   │
   ▼
PreprocessedAudio (Canonical 16kHz Mono float32 PCM Array)
   │
   ▼
[4. Feature Extraction] (AudioFeatureExtractor)
   ├─ Log-Mel Spectrogram (80 bins)
   ├─ MFCC (20 coefficients)
   ├─ Spectral Centroid (mean/std/delta)
   └─ Spectral Rolloff (85% energy cutoff)
   │
   ├─────────────────────────────────────────┐
   ▼                                         ▼
[5. Model Inference]                     [6. Visualization Payload]
(ModelRegistry -> BaselineClassifier)     (VisualizationBuilder)
   │ Status: "analysis_only"                 ├─ 300-point min/max peak envelope
   │ Probabilities: null (Uncalibrated)      └─ 40x120 normalized [0, 1] grid
   │
   └───────────────────┬─────────────────────┘
                       ▼
[7. Forensic Report Assembly] (ForensicReportBuilder)
   ├─ Audio metadata summary
   ├─ Acoustic observations
   ├─ Evidence indicators with explicit provenance tags
   └─ Mandatory evaluation disclaimer
                       │
                       ▼
              HTTP 200 JSON Response
```

---

## 3. REST API Specification

### Endpoints Overview

| Method | Endpoint | Description | Status Codes |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health & operational status | `200` |
| `POST` | `/api/v1/analyze` | End-to-end audio analysis pipeline | `200`, `400`, `413`, `415`, `422`, `500` |

---

### A. Health Check Endpoint

- **Endpoint**: `GET /api/v1/health`
- **Response**:
  ```json
  {
    "status": "ok",
    "service": "voxguard-api",
    "version": "0.1.0"
  }
  ```

---

### B. Audio Analysis Endpoint

- **Endpoint**: `POST /api/v1/analyze`
- **Request Format**: `multipart/form-data`
  - Form parameter `file`: Raw binary audio file (`.wav`, `.mp3`, `.flac`, `.ogg`). Max size 10 MB.

- **Successful Response Structure (`HTTP 200 OK`)**:
  ```json
  {
    "success": true,
    "audio_metadata": {
      "duration_seconds": 4.0,
      "sample_rate": 16000,
      "channels": 1,
      "num_frames": 64000,
      "metadata": {
        "original_sample_rate": 44100,
        "original_channels": 2,
        "resampled": true,
        "channel_converted": true,
        "peak_scaling_factor": 0.85,
        "dtype": "float32"
      }
    },
    "classification": {
      "label": "uncertain",
      "probabilities": {
        "human": null,
        "synthetic": null
      },
      "confidence_score": null,
      "model_name": "VoxGuard Statistical Acoustic Baseline",
      "model_version": "0.1.0-baseline",
      "status": "analysis_only",
      "metadata": {
        "execution_time_ms": 3.42,
        "device": "cpu",
        "disclaimer": "All values are descriptive acoustic indicators, not a validated deepfake classification.",
        "acoustic_statistics": {
          "spectral_centroid_mean_hz": 1420.55,
          "spectral_centroid_std_hz": 312.4,
          "spectral_rolloff_mean_hz": 5150.0,
          "high_band_mel_mean_db": -48.2,
          "low_band_mel_mean_db": -22.1,
          "mfcc_mean": -3.85,
          "mfcc_variance": 42.1
        }
      }
    },
    "visualization": {
      "waveform": {
        "target_points": 300,
        "min_peaks": [-0.85, -0.42, "..."],
        "max_peaks": [0.91, 0.45, "..."],
        "peak_envelope": [0.91, 0.45, "..."],
        "duration_seconds": 4.0
      },
      "spectrogram": {
        "freq_bins": 40,
        "time_bins": 120,
        "data_grid": [[0.12, 0.45, "..."], "..."],
        "min_db": -80.0,
        "max_db": 0.0
      }
    },
    "forensic_report": {
      "audio_metadata": {
        "duration_seconds": 4.0,
        "sample_rate_hz": 16000,
        "channels": 1,
        "num_frames": 64000,
        "format": "wav"
      },
      "acoustic_observations": {
        "spectral_centroid": { "mean_hz": 1420.55, "std_hz": 312.4 },
        "spectral_rolloff_85_percent": { "mean_hz": 5150.0, "std_hz": 420.0 },
        "mfcc_statistics": { "num_coefficients": 20, "mean": -3.85, "variance": 42.1 },
        "signal_characteristics": { "average_energy_db": -24.5, "dynamic_range_db": 68.2 }
      },
      "model_assessment": null,
      "evidence_indicators": [
        {
          "name": "Sample Rate Standardization",
          "measured_value": "16000 Hz Mono",
          "interpretation": "Audio stream decoded and standardized to canonical 16 kHz mono format.",
          "severity": "INFO",
          "confidence_strength": "N/A",
          "is_model_derived": false,
          "provenance": "metadata"
        },
        {
          "name": "Spectral Centroid Distribution",
          "measured_value": "1420.6 Hz",
          "interpretation": "Observed acoustic characteristic representing the brightness center of spectral mass.",
          "severity": "INFO",
          "confidence_strength": "N/A",
          "is_model_derived": false,
          "provenance": "signal_analysis"
        },
        {
          "name": "System Operational Mode",
          "measured_value": "Analysis Only (Baseline)",
          "interpretation": "System operating on uncalibrated statistical baseline. No ML detection probabilities computed.",
          "severity": "INFO",
          "confidence_strength": "N/A",
          "is_model_derived": false,
          "provenance": "heuristic"
        }
      ],
      "disclaimer": "VoxGuard forensic reports provide acoustic feature indicators and model assessments. Acoustic indicators reflect observed signal characteristics and do not constitute proof of AI-generated speech. Refer to docs/EVALUATION_METRICS.md for benchmarking protocols.",
      "timestamp": "2026-09-23T19:51:48.123456+00:00"
    }
  }
  ```

---

## 4. Structured Error Contract & HTTP Status Codes

All API errors return a standard JSON error envelope without exposing internal stack traces or server file paths:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE_STRING",
    "message": "Human readable error description."
  }
}
```

### HTTP Status Code Mapping Matrix

| Status Code | Code String | Condition |
| :--- | :--- | :--- |
| **`400 Bad Request`** | `MISSING_INPUT` | Request form contains no audio file payload |
| **`400 Bad Request`** | `EMPTY_FILE` | Uploaded audio file is 0 bytes |
| **`413 Payload Too Large`** | `FILE_TOO_LARGE` | Upload exceeds max configured size (10 MB) |
| **`415 Unsupported Media Type`** | `UNSUPPORTED_EXTENSION` | File extension is not in permitted list (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.webm`, `.opus`, `.wma`, `.aiff`, `.caf`, `.amr`, `.3gp`, `.mp4`) |
| **`415 Unsupported Media Type`** | `UNSUPPORTED_MIME_TYPE` | Content-type header is not in permitted list |
| **`422 Unprocessable Entity`** | `CORRUPTED_AUDIO` | Magic header corrupted or container format mismatch |
| **`422 Unprocessable Entity`** | `AUDIO_DURATION_EXCEEDED` | Decoded duration exceeds maximum allowed limit (300s) |
| **`422 Unprocessable Entity`** | `DECODING_FAILED` | Audio stream frames corrupted or undecodable |
| **`500 Internal Error`** | `INTERNAL_SERVER_ERROR` | Unexpected internal exception |

---

## 5. Security & Resource Safeguards

1. **In-Memory Streaming**: Files are read directly from stream memory buffers. No temp files linger on disk.
2. **Path Traversal Protection**: Filename paths are stripped using `Path(filename).name`.
3. **No Unsanitized Execution**: Passive byte decoding only.
4. **CORS Policy**: Configured explicitly via `ALLOWED_ORIGINS` environment variable (e.g. `http://localhost:3000`). Unrestricted `*` origins are disabled in production.

---

## 6. Dataset Preparation & Validation Pipeline

VoxGuard includes an offline dataset preparation and validation engine that validates manually supplied audio datasets and generates deterministic speaker-disjoint partitions prior to model training.

### Pipeline Flow

```
Audio Dataset Directory
       │
       ▼
Manifest CSV (manifest.csv)
       │
       ▼
[1. ManifestValidator] ───► Verifies CSV schema, header columns, label/split values, & path safety
       │
       ▼
[2. Audio File Validation] ──► Audits local file existence & decodes audio stream metadata (duration, format)
       │
       ▼
[3. SpeakerSplitValidator] ──► Checks 0% speaker leakage across train, val, test splits
       │
       ▼
[4. ClassBalanceAnalyzer] ──► Computes human vs. synthetic ratios & generator source distributions
       │
       ▼
[5. DeterministicSpeakerSplitter] ──► Shuffles & partitions unassigned manifests by speaker ID (seed 42)
       │
       ▼
[6. DatasetReportBuilder] ──► Consolidates statistics into structured JSON / CLI audit report
       │
       ▼
Validated Training & Evaluation Input
```

### Pipeline Components & Abstractions:

1. **`ManifestValidator` (`backend/app/evaluation/manifest_validator.py`)**:
   - Validates CSV headers (`file_path`, `label`, `speaker_id`, `source`, `source_id`, `split`).
   - Ensures `label` is strictly `human` or `synthetic`.
   - Ensures `split` is strictly `train`, `validation`, `test`, or `ood`.
   - Sanitizes `file_path` to block directory traversal (`../`).

2. **`SpeakerSplitValidator` (`backend/app/evaluation/speaker_split_validator.py`)**:
   - Computes set intersections of `speaker_id` across splits.
   - Enforces $S_{\text{train}} \cap S_{\text{val}} = \emptyset$, $S_{\text{train}} \cap S_{\text{test}} = \emptyset$, $S_{\text{val}} \cap S_{\text{test}} = \emptyset$.

3. **`ClassBalanceAnalyzer` (`backend/app/evaluation/class_balance.py`)**:
   - Tracks counts and percentage ratios for `human` vs. `synthetic` classes globally and per split.
   - Summarizes audio samples per generator `source` (e.g. ElevenLabs, VALL-E, Tacotron2).

4. **`DatasetReportBuilder` (`backend/app/evaluation/dataset_report.py`)**:
   - Integrates manifest validation, audio file auditing, speaker leakage validation, and class balance reporting.
   - Outputs a consolidated `DatasetReport` containing min, max, mean, median audio duration statistics and overall readiness status (`READY`, `WARNING`, `INVALID`).

5. **`DeterministicSpeakerSplitter` (`backend/app/evaluation/splitter.py`)**:
   - Groups dataset rows by `speaker_id` before shuffling with a fixed seed (default `42`).
   - Allocates full speaker clusters into target split ratios (default 70/15/15) to guarantee zero speaker leakage.
   - Preserves pre-designated `ood` split samples.

---

## 7. Feature Dataset Generation Pipeline

> [!IMPORTANT]
> **Phase 11 Scope**: Phase 11 creates reproducible feature dataset artifacts (`.npz`) and checksum metadata (`features_manifest.json`) from validated local audio files. **Phase 11 does NOT train a model.**

### Pipeline Flow

```
Validated Audio Dataset & Manifest CSV
                  │
                  ▼
       [1. Audio Validation] (AudioValidator - path safety, size, MIME, magic bytes)
                  │
                  ▼
     [2. Canonical Preprocessing] (AudioPreprocessor - 16kHz mono float32 PCM)
                  │
                  ▼
      [3. Feature Extraction] (AudioFeatureExtractor - Log-Mel, MFCC, Centroid, Rolloff)
                  │
                  ▼
       [4. Feature Validation] (FeatureValidator - float32, finiteness, shapes)
                  │
                  ▼
       [5. Feature Artifact] (.npz files in split subdirectories: train, val, test, ood)
                  │
                  ▼
     [6. Offline Training Input] (Future Offline Training Pipeline)
```

### Key Components:

1. **`DatasetIngestor` (`backend/app/evaluation/dataset_ingestor.py`)**: Sequentially reads manifest rows, validates audio, decodes PCM streams, converts to 16kHz mono float32, extracts feature tensors, and yields `DatasetSample` records without loading the entire dataset into RAM.
2. **`FeatureValidator` (`backend/app/evaluation/feature_validator.py`)**: Audits feature arrays for `float32` dtype, expected tensor ranks, non-emptiness, finiteness (no NaN or Inf), and temporal frame alignment.
3. **`DatasetIntegrity` (`backend/app/evaluation/dataset_integrity.py`)**: Computes SHA-256 digests for generated `.npz` files and outputs sanitized `features_manifest.json`.
4. **`FeatureDatasetBuilder` (`backend/app/evaluation/feature_builder.py`)**: Orchestrates end-to-end dataset generation, split isolation re-verification, `.npz` artifact writing, `processing_report.json`, `features_manifest.csv`, and `statistics.json` generation.

---

---

## 8. Model Training & Evaluation Infrastructure

### Pipeline Flow

```
Compressed NPZ Feature Artifacts (.npz)
                  │
                  ▼
         [1. FeatureDataset] ──► Path safety, split filter, NPZ validation (allow_pickle=False)
                  │
                  ▼
        [2. FeatureWindowing] ──► Converts variable T into 256-frame windows (stride 128, zero-padded)
                  │
                  ▼
   [3. TrainingBatchValidator] ──► Audits array shapes, float32 dtype, finiteness (no NaN/Inf), & labels
                  │
                  ▼
     [4. TrainingBatchCollator] ──► Batches samples into 3D/1D NumPy arrays & preserves metadata
                  │
                  ▼
     [5. TrainingDataLoader] ──► Deterministic seeded sampler (shuffle=True for train, False for val/test)
                  │
                  ▼
  [6. VoxGuardAcousticNet] ──► 1D CNN (100 in_channels, 3 Conv blocks + BatchNorm + Dropout, 2-class FC)
                  │
                  ▼
         [7. Trainer Engine] ──► Class weighting, AdamW, early stopping, best_model.pt & manifest.json
                  │
                  ▼
      [8. ModelEvaluator] ──► Accuracy, F1, ROC-AUC, PR-AUC, Confusion Matrix, threshold tuning
```

### Exact Batch Tensor Shapes & Specifications:
- **`log_mel`**: `(B, 80, 256)`, `float32`
- **`mfcc`**: `(B, 20, 256)`, `float32`
- **`spectral_centroid`**: `(B, 1, 256)`, `float32`
- **`spectral_rolloff`**: `(B, 1, 256)`, `float32`
- **`labels`**: `(B,)`, `int64` (`0` = human, `1` = synthetic)
- **`metadata`**: List of $B$ dictionaries preserving `sample_id`, `window_index`, `speaker_id`, `source`, `source_id`, `split`, `label_str`.

### Key Components:
1. **`VoxGuardAcousticNet` (`backend/app/training/model.py`)**: 1D Convolutional Neural Network operating on 100-channel concatenated Log-Mel & MFCC feature tensors with adaptive pooling and 2-class linear head.
2. **`DatasetStatisticsCalculator` (`backend/app/training/statistics.py`)**: Calculates window/sample/speaker distributions, sequence length stats, and audits dataset risk flags (class imbalance, dominant speaker/source, suspicious split size).
3. **`Trainer` (`backend/app/training/trainer.py`)**: PyTorch training engine supporting class-weighted loss, AdamW optimizer, gradient clipping, early stopping on validation metrics, saving `best_model.pt` / `last_model.pt`, and automatic `manifest.json` SHA-256 digest creation.
4. **`ModelEvaluator` (`backend/app/evaluation/model_evaluator.py`)**: Evaluates model performance across splits computing Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrices.
5. **Training Status Endpoint (`GET /api/v1/training/status`)**: Exposes deterministic training infrastructure readiness, total window counts, speaker counts, zero leakage status, and dataset risk assessment to the frontend dashboard.
6. **CLI Tools (`scripts/train_model.py`, `scripts/evaluate_model.py`)**: Offline command line interfaces for reproducible training execution, validation threshold tuning, dry-run sanity checks with 5 output JSON artifacts (`dataset_summary.json`, `training_config.json`, `feature_schema.json`, `split_summary.json`, `integrity.json`).




