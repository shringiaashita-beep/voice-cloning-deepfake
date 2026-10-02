# VoxGuard: AI Audio Deepfake Detection & Forensic Analysis Platform

VoxGuard is an open-source, AI-powered platform designed to detect synthetic/deepfake speech, analyze audio authenticity, and generate explainable forensic reports.

---

## 🌟 Key Features

- **Voice Clone Deepfake Detection (Active Neural Forensics)**: Classifies speech recordings into Genuine Human, AI-Generated / Cloned Voice, or Uncertain with calibrated probability scores.
- **Multi-Architecture Voice Clone Fingerprinting**: Detects acoustic signatures characteristic of state-of-the-art voice synthesis:
  - **ElevenLabs / Neural Vocoders (HiFi-GAN / BigVGAN)**: 7.5 kHz spectral boundary dropoff and pitch micro-tremor suppression.
  - **RVC / So-VITS**: Pitch quantization step jumps and formant phase warping.
  - **Diffusion / Autoregressive TTS (Bark / XTTS)**: Frame-level spectral flux dithering and synthetic pause floor.
  - **Natural Biological Vocal Tract**: Organic micro-jitter (0.6%–1.8%), rich 20-band MFCC resonance, and natural aspiration.
- **Deepfake Risk Gauge & Probability Meter**: Radial visual meter displaying real-time synthetic likelihood (0% to 100%) and calibrated confidence.
- **Explainable Forensic Report & Audit Certificate**: Explains vocoder cutoffs, pitch jitter metrics, and enables 1-click JSON forensic audit certificate export.
- **Interactive Preset Test Bench**: Instant one-click testing against simulated ElevenLabs, RVC, Bark, and Authentic Human voice recordings.
- **Strict Audio Validation**: Securely validates MIME types, file headers, duration, and file size limits (Max 10 MB: `.wav`, `.mp3`, `.flac`, `.ogg`).
- **Standardized Backend Processing**: Centralized Python-based decoding, 16 kHz mono resampling, and loudness normalization.
- **Visual Analytics**: Server-optimized waveform peak envelopes and spectrogram visualizations.

---

## 🏗️ Architecture

```
Frontend (Next.js, React, Tailwind CSS)
   │
   ▼ REST API (JSON)
Backend API Layer (FastAPI)
   │
   ▼
Audio Processing (Validation, Decoding, Resampling, Normalization)
   │
   ▼
Feature Extraction (Spectrograms, MFCCs, Spectral Centroid, Phase Coherence)
   │
   ├───────────────────────────────┐
   ▼                               ▼
ML Model Inference Engine    Visualization Formatter (Downsampled Bins)
   │                               │
   ▼                               │
Probability Calibration            │
   │                               │
   ▼                               │
Forensic Report Generator ◄────────┘
   │
   ▼
Frontend Presentation
```

For full architectural details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
For model evaluation metrics and benchmarking guidelines, see [docs/EVALUATION_METRICS.md](docs/EVALUATION_METRICS.md).

---

## 📁 Repository Structure

```
VoxGuard/
│
├── frontend/                  # Next.js frontend application
│   ├── src/
│   │   ├── app/               # App Router pages
│   │   ├── components/        # React components (AudioUploader, Report, Visualizers)
│   │   └── types/             # TypeScript API interfaces
│   ├── package.json
│   └── tailwind.config.js
│
├── backend/                   # FastAPI Python application
│   ├── app/
│   │   ├── main.py            # FastAPI entrypoint & routes setup
│   │   ├── config.py          # App configuration & environment variables
│   │   ├── audio/             # Audio validation, decoding, resampling & normalization
│   │   ├── features/          # Feature extraction & visualization builders
│   │   ├── models/            # ML classifier runner & probability calibration
│   │   ├── forensics/         # Forensic report builder & indicator rules
│   │   └── routes/            # REST API endpoint handlers
│   ├── tests/                 # Unit & integration test suite
│   ├── requirements.txt       # Python dependencies list
│   └── .env.example           # Environment template
│
├── docs/                      # Technical documentation & evaluation guidelines
├── models/                    # Model weight artifacts (gitignored)
├── datasets/                  # Datasets directory (gitignored)
├── notebooks/                 # Analysis and experimentation notebooks
└── audio_temp/                # Temporary processing buffer directory
```

---

## 🚀 Setup Instructions (Manual Review & Run)

### Backend Setup (Python)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. Review `requirements.txt` and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment configuration:
   ```bash
   cp .env.example .env
   ```
5. Run the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

### Frontend Setup (Next.js)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Review `package.json` and install dependencies:
   ```bash
   npm install
   ```
3. Run the Next.js development server:
   ```bash
   npm run dev
   ```
4. Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🤖 ML Classifier & Model Configuration

VoxGuard supports pluggable ML Deepfake Classifiers (`RealMLClassifier`) with zero automatic downloads or weight fabrications.

### Model Environment Variables (`.env`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MODEL_ENABLED` | `False` | Master switch to activate real ML deepfake classifier adapter. |
| `MODEL_PATH` | `""` | File path to on-disk model weights (`.pt`, `.onnx`). |
| `MODEL_NAME` | `"VoxGuard Neural Deepfake Classifier"` | Human-readable model identifier. |
| `MODEL_VERSION` | `"1.0.0"` | Semantic version string of active model. |
| `MODEL_TYPE` | `"pytorch"` | Framework adapter (`pytorch`, `torchscript`, `onnx`). |

### Analysis-Only Fallback Policy

- If `MODEL_ENABLED=False` or `MODEL_PATH` points to a missing file:
  - VoxGuard automatically operates in **`ANALYSIS_ONLY`** or **`MODEL_NOT_READY`** mode.
  - Probabilities (`human`, `synthetic`) are set strictly to `null` (`N/A`).
  - No acoustic heuristic is converted into fake probabilities.

### Model Evaluation Safety Protocol

Before any trained model reaches `status = "ready"`, it must be benchmarked on:
1. **Zero-Speaker-Overlap Splits**: Partitioned strictly by Speaker ID between train/val/test sets.
2. **Standard Evaluation Metrics**: EER (Equal Error Rate), ROC-AUC, F1-Score, Precision, Recall, FPR, FNR.
3. **Out-of-Distribution (OOD) Testing**: Benchmarked on unseen neural speech generators (ElevenLabs, VALL-E) and lossy codecs (MP3/OGG).
For complete benchmarking protocols, refer to [docs/EVALUATION_METRICS.md](docs/EVALUATION_METRICS.md).

---

## 📊 Dataset Preparation & Validation

VoxGuard provides dataset validation and deterministic speaker-disjoint splitting tools to ensure offline datasets are validated and free of speaker leakage prior to model training.

### Dataset Directory & Manifest Schema

Store datasets locally in `datasets/` with a manifest CSV file (`manifest.csv`). The CSV must follow this schema:

| Column | Type | Description |
| :--- | :--- | :--- |
| `file_path` | String | Path to audio file relative to base directory (e.g. `datasets/wavs/sample01.wav`). |
| `label` | String | Audio class label: `human` or `synthetic`. |
| `speaker_id` | String | Unique identifier for the speaker source (required for leakage validation). |
| `source` | String | Generator/Dataset provenance name (e.g. `LibriSpeech`, `ElevenLabs`, `VALL-E`). |
| `source_id` | String | Identifier for specific voice model or recording session. |
| `split` | String | Dataset split: `train`, `validation`, `test`, or `ood` (out-of-distribution). |

An example manifest template is provided at `datasets/manifest.example.csv`.

### Speaker-Disjoint Partitioning Requirement

To prevent data leakage, **no `speaker_id` may exist in more than one partition** among `train`, `validation`, and `test`. VoxGuard enforces zero speaker overlap to guarantee reliable generalization benchmarks.

### Dataset Validation CLI

Audit dataset manifest CSV syntax, audio stream decodability, duration metrics, class balance, and speaker leakage without downloading data or running external APIs:

```bash
python scripts/validate_dataset.py --manifest datasets/manifest.csv --base-dir .
```

### Deterministic Speaker Splitter CLI

Partition an unassigned or legacy manifest into zero-speaker-overlap train/val/test splits using deterministic seed shuffling (default seed `42`):

```bash
python scripts/split_dataset.py --manifest datasets/manifest.csv --output datasets/manifest_prepared.csv --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

For full details on dataset preparation, schema validation, and guidelines, see [docs/DATASET_GUIDE.md](docs/DATASET_GUIDE.md).

---

## 🎛️ Feature Dataset Generation

VoxGuard converts validated local audio manifests into ML-ready feature artifacts (`.npz`) containing Log-Mel spectrograms (80 bins), MFCCs (20 coefficients), Spectral Centroid (1), and Spectral Rolloff (1).

### Feature Generation CLI

To build feature dataset artifacts:

```bash
# Dry-run validation (validates manifest, paths, & speaker separation without generating files)
python scripts/build_features.py --manifest datasets/manifest.csv --dry-run

# Generate feature dataset artifacts (.npz) into output directory
python scripts/build_features.py --manifest datasets/manifest.csv --output datasets/prepared/features
```

For full details on feature dimensions, `.npz` artifact schemas, and pipeline specifications, see [docs/FEATURE_DATASET_GUIDE.md](docs/FEATURE_DATASET_GUIDE.md).

---

## ⚙️ Model Training & Evaluation Infrastructure

VoxGuard provides offline model training (`VoxGuardAcousticNet`), 256-frame sliding feature windowing, dataset statistics risk assessment, early stopping, class-weighted cross-entropy loss, validation decision threshold tuning, and benchmark evaluation across train, validation, test, and OOD splits.

For complete training pipeline documentation, windowing specs, and dry-run guidelines, see [docs/TRAINING_PIPELINE_GUIDE.md](docs/TRAINING_PIPELINE_GUIDE.md).

### Offline Model Training CLI

To train a model checkpoint on pre-extracted feature artifacts and generate SHA-256 `manifest.json`:

```bash
python scripts/train_model.py \
  --manifest datasets/prepared/features_manifest.csv \
  --feature-dir datasets/prepared/features \
  --output-dir models/trained \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --patience 5
```

### Model Evaluation & Threshold Tuning CLI

To evaluate a trained checkpoint across splits (`validation`, `test`, `ood`), tune decision threshold on validation split, and save JSON evaluation report:

```bash
python scripts/evaluate_model.py \
  --model-path models/trained/best_model.pt \
  --manifest datasets/prepared/features_manifest.csv \
  --feature-dir datasets/prepared/features \
  --splits validation,test,ood \
  --threshold-search \
  --output-report reports/evaluation_summary.json
```

For complete details on model architecture, training configuration, and metrics, see [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md).

---

## 📄 License & Safety Notice

VoxGuard is created for academic and security research purposes. All probability outputs represent statistical predictions from evaluated ML models and must be interpreted alongside forensic indicator reports.

