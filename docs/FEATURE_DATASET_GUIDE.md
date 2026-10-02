# VoxGuard Feature Dataset Generation Guide

This guide describes the reproducible dataset-ingestion and feature-generation pipeline in VoxGuard. The feature dataset pipeline transforms validated audio manifests into ML-ready numerical feature artifacts (`.npz`) without training models or downloading external data.

---

## 🏗️ Feature Dataset Pipeline Overview

```
Validated Manifest CSV (manifest.csv)
       │
       ▼
Audio File Path Resolution (Sanitized relative paths)
       │
       ▼
[AudioValidator] ──► MIME, magic header, duration, size checks
       │
       ▼
[AudioDecoder] ──► Decodes raw audio stream to PCM float32 array
       │
       ▼
[AudioPreprocessor] ──► 16,000 Hz Mono peak-normalized [-1.0, 1.0] PCM array
       │
       ▼
[AudioFeatureExtractor] ──► Extracts Log-Mel (80), MFCC (20), Centroid (1), Rolloff (1)
       │
       ▼
[FeatureValidator] ──► Audits dtype (float32), shapes, finiteness (no NaN/Inf), temporal alignment
       │
       ▼
[DatasetIntegrity] ──► Computes SHA-256 artifact digests
       │
       ▼
ML Feature Artifacts (.npz) & Manifest Metadata JSON
```

---

## 📐 Feature Specifications & Tensor Dimensions

| Feature Name | Axis 0 (Frequency/Bins) | Axis 1 (Temporal Frames $T$) | Data Type | Formula / Method |
| :--- | :--- | :--- | :--- | :--- |
| **Log-Mel Spectrogram** | 80 Mel frequency bands | $T$ frames ($T > 0$) | `float32` | $10 \log_{10}(\text{MelFilterbank} \cdot |\text{STFT}|^2)$ |
| **MFCC** | 20 coefficients | $T$ frames ($T > 0$) | `float32` | $\text{DCT-II}(\text{Log-Mel})$ |
| **Spectral Centroid** | 1 frequency center | $T$ frames ($T > 0$) | `float32` | $\frac{\sum f \cdot S(f)}{\sum S(f)}$ |
| **Spectral Rolloff** | 1 frequency boundary | $T$ frames ($T > 0$) | `float32` | 85% spectral energy threshold |

### DSP Configuration Parameters:
- **Sample Rate**: 16,000 Hz
- **FFT Size (`n_fft`)**: 1,024 (~64 ms)
- **Hop Length (`hop_length`)**: 512 (~32 ms, 50% window overlap)
- **Mel Bands (`n_mels`)**: 80
- **MFCCs (`n_mfcc`)**: 20
- **Feature Pipeline Version**: `"voxguard-features-0.1.0"`

---

## 🏷️ Label Encoding

Labels are encoded deterministically strictly from the validated CSV manifest (never inferred from filenames):

| String Label | Integer Code |
| :--- | :--- |
| `human` | `0` |
| `synthetic` | `1` |

Encoding functions are provided in `backend/app/evaluation/labels.py`:
- `encode_label("human") -> 0`
- `decode_label(0) -> "human"`

---

## 📁 Artifact Format & Storage Layout

Generated feature artifacts are stored in split subdirectories under `datasets/prepared/features/`:

```
datasets/prepared/features/
├── train/
│   ├── train_spk001_sample001.npz
│   └── ...
├── validation/
│   ├── validation_spk003_sample003.npz
│   └── ...
├── test/
│   ├── test_spk004_sample004.npz
│   └── ...
├── ood/
│   ├── ood_spk104_sample005.npz
│   └── ...
├── features_manifest.csv
├── features_manifest.json
├── processing_report.json
└── statistics.json
```

### Compressed NumPy `.npz` Schema

Each `.npz` file contains the following arrays:
- `log_mel`: `(80, T)` `float32` array
- `mfcc`: `(20, T)` `float32` array
- `spectral_centroid`: `(1, T)` `float32` array
- `spectral_rolloff`: `(1, T)` `float32` array
- `label_code`: `int64` scalar (0 or 1)
- `sample_id`, `speaker_id`, `source`, `source_id`, `split`: string metadata scalars

---

## ⏱️ Variable-Length Audio & Temporal Frames

Audio files in the dataset have varying durations. Feature frame sequence lengths ($T$) are kept variable-length at this stage. Frame counts are recorded per sample (`num_feature_frames`). Padding, cropping, or windowing is handled separately during training batch creation.

---

## 🔒 Split Isolation & Speaker Leakage Safeguards

Before feature generation begins, `SpeakerSplitValidator` re-audits speaker disjointness across `train`, `validation`, and `test` splits. If speaker leakage ($S_{\text{train}} \cap S_{\text{val}} \neq \emptyset$) is detected, feature dataset generation aborts immediately.

---

## 🛡️ Processing Failures & Integrity Metadata

- **`features_manifest.json`**: Contains dataset version, pipeline version, UTC generation timestamp, sample count, and relative SHA-256 hashes for all generated `.npz` files.
- **`processing_report.json`**: Records counts of successful vs. failed samples. Corrupt or unreadable audio samples produce clear failure records with safe error codes and messages without throwing stack traces or exposing server paths.
- **`statistics.json`**: Records actual calculated duration and feature frame statistics (min, max, mean, median) per split.

---

## 💻 CLI Usage

### Dry-Run Validation

Validate manifest CSV syntax, file paths, and speaker disjointness without generating files:

```bash
python scripts/build_features.py --manifest datasets/manifest.csv --dry-run
```

### Feature Dataset Generation

Generate feature artifacts into `datasets/prepared/features/`:

```bash
python scripts/build_features.py --manifest datasets/manifest.csv --output datasets/prepared/features
```

### Overwrite Existing Artifacts

Permit replacing existing output directory contents:

```bash
python scripts/build_features.py --manifest datasets/manifest.csv --output datasets/prepared/features --overwrite
```

> [!NOTE]
> **Safety Notice**: VoxGuard Phase 11 generates ML feature artifacts only. No model training, downloading, or external network requests occur during feature dataset generation.

---

## ⚙️ Training Data DataLoader & Batch Preparation

Phase 12 introduces fixed 256-frame windowing, batch collation, and deterministic sampling for ML training loaders (`FeatureDataset`, `FeatureWindowing`, `TrainingDataLoader`).

### Windowing & Padding Strategy
- **Target Frames (`TARGET_FRAMES`)**: 256 frames (~8.192 seconds at 16kHz with 512 hop length).
- **Window Stride (`WINDOW_STRIDE`)**: 128 frames (50% window overlap for $T > 256$).
- **Padding (`PADDING_MODE`)**: Right-side zero padding for sequences $T < 256$.
- **Tail Window Handling**: For $T > 256$, if the tail length is not an exact multiple of the stride, a final right-aligned window `[T - 256 : T]` is extracted to preserve all tail audio information deterministically.

### Speaker Isolation & Metadata Preservation
During collation, batch metadata retains full sample provenance (`speaker_id`, `source`, `source_id`, `sample_id`, `window_index`, `split`, `label_str`). No speaker data leakage occurs across split boundaries.

### Label Mapping
- `human` = `0`
- `synthetic` = `1`

### Data Inspection CLI Usage

Inspect generated feature datasets and batch tensor dimensions without executing model training:

```bash
python scripts/inspect_training_data.py --manifest datasets/prepared/features_manifest.csv --split train
```

