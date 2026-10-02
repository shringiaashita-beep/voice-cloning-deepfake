# VoxGuard Dataset Preparation & Validation Guide

This document specifies the standards, manifest formats, validation protocols, and CLI commands for preparing audio datasets in VoxGuard.

> [!IMPORTANT]
> **Safety Notice**: VoxGuard does NOT automatically download audio datasets or external speech corpora. Researchers and system administrators must manually obtain audio datasets legally and place them in the `datasets/` directory.

---

## 1. Dataset Directory Structure

```
datasets/
├── README.md                  # Guidelines and quick-start instructions
├── manifest.example.csv      # Example CSV manifest template
├── raw/                       # Place raw un-split audio files here
│   ├── human/                # Genuine human speech recordings (.wav, .mp3, .flac, .ogg)
│   └── synthetic/            # AI-generated speech recordings (.wav, .mp3, .flac, .ogg)
└── prepared/                  # Destination for prepared speaker-disjoint manifests
```

---

## 2. CSV Manifest Specifications

Manifests must be UTF-8 encoded CSV files containing the following required header columns:

```csv
file_path,label,speaker_id,source,source_id,split
raw/human/spk01_01.wav,human,spk_001,librispeech,libri_reader_1,train
raw/human/spk01_02.wav,human,spk_001,librispeech,libri_reader_1,train
raw/synthetic/gen01_01.wav,synthetic,spk_101,elevenlabs,v2_neural,validation
```

### Column Specification:

| Column Name | Allowed Values / Format | Description |
| :--- | :--- | :--- |
| `file_path` | Dataset-relative path | Path relative to `datasets/` root. Must not use absolute paths or `../` path traversal. |
| `label` | `human`, `synthetic` | Binary classification decision label. |
| `speaker_id` | `string` | Unique identifier of the speaker (e.g. `spk_001`). Mandatory for speaker-disjoint splitting. |
| `source` | `string` | Dataset or origin category (e.g., `librispeech`, `asvspoof`, `elevenlabs`). |
| `source_id` | `string` | Generator model or specific speaker source variant (e.g., `libri_reader_1`, `bark_v1`). |
| `split` | `train`, `validation`, `test`, `ood` | Target split partition. |

---

## 3. Mandatory Speaker-Disjoint Split Requirement

To prevent acoustic data leakage and memorization of individual speaker voice characteristics:

1. **Zero Speaker Overlap**: A `speaker_id` present in `train` MUST NOT appear in `validation`, `test`, or `ood`.
2. **Speaker-Level Partitioning**: Splitting must partition entire speaker groups, **never** individual audio files belonging to the same speaker.
3. **Out-of-Distribution (OOD) Testing**: OOD splits contain unseen neural speech generators or audio codecs to benchmark model generalization.

---

## 4. CLI Commands & Workflow

### A. Validate Dataset Readiness
Run `scripts/validate_dataset.py` to audit CSV formatting, audio file decodability, speaker leakage, and class balance:

```bash
python scripts/validate_dataset.py --manifest datasets/manifest.example.csv
```

### B. Generate Deterministic Speaker-Disjoint Split
Run `scripts/split_dataset.py` to deterministically partition a raw manifest by Speaker ID without speaker leakage:

```bash
python scripts/split_dataset.py \
    --manifest datasets/manifest.example.csv \
    --output datasets/prepared/manifest.csv \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
```
