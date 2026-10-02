# VoxGuard Datasets Directory

This directory is designated for storing local audio datasets, manifest CSV files, and prepared speaker-disjoint dataset partitions.

> [!IMPORTANT]
> **Safety Notice**: VoxGuard does not automatically download audio datasets or external corpora. All audio files must be obtained legally and placed here manually by researchers or system operators.

---

## 📁 Recommended Directory Structure

```
datasets/
├── README.md                  # Documentation and guidelines (this file)
├── manifest.example.csv      # Example CSV manifest template
├── raw/                       # Place raw un-split audio files here
│   ├── human/                # Genuine human speech recordings (.wav, .mp3, .flac, .ogg)
│   └── synthetic/            # AI-generated speech recordings (.wav, .mp3, .flac, .ogg)
└── prepared/                  # Destination for prepared speaker-disjoint manifests
```

---

## 📄 Manifest Format Specifications

Dataset manifests must be formatted as UTF-8 encoded CSV files containing the following required header columns:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `file_path` | `string` | Relative path from `datasets/` root (e.g. `raw/human/spk01_01.wav`). Must not use unsafe `../` path traversal. |
| `label` | `enum` | Must be either `human` or `synthetic`. |
| `speaker_id` | `string` | Unique identifier of the speaker (e.g. `spk_001`). Mandatory for speaker-disjoint splitting. |
| `source` | `string` | Origin dataset or primary category (e.g. `librispeech`, `asvspoof`, `elevenlabs`). |
| `source_id` | `string` | Specific generator or speaker source variant (e.g. `libri_reader_1`, `v2_neural`). |
| `split` | `enum` | Target partition: `train`, `validation`, `test`, or `ood`. |

---

## 🛠️ Dataset Validation & Splitting CLI Usage

VoxGuard provides standalone scripts to audit dataset integrity and generate speaker-disjoint splits:

### 1. Validate Dataset Integrity:
```bash
python scripts/validate_dataset.py --manifest datasets/manifest.example.csv
```

### 2. Generate Speaker-Disjoint Split:
```bash
python scripts/split_dataset.py \
    --manifest datasets/manifest.csv \
    --output datasets/prepared/manifest.csv \
    --seed 42
```
