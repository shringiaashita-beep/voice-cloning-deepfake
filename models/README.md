# VoxGuard Model Artifacts Directory

This directory is designated for storing manually supplied, externally trained deepfake audio classification model checkpoints (`.pt`, `.onnx`) and JSON manifests.

> [!IMPORTANT]
> **Safety Policy**: VoxGuard does not automatically download model weights or datasets from external networks. Model artifacts must be placed here manually by security researchers or system administrators.

---

## 📁 Expected Directory Structure

```
models/
├── README.md                  # Instructions and schema guidelines (this file)
├── manifest.example.json      # Standard JSON manifest template
└── voxguard_model.pt         # (Optional) Manually supplied trained model checkpoint
```

---

## 🛠️ Step-by-Step Instructions to Supply a Local Model

1. **Place Model Checkpoint**:
   Copy your trained PyTorch (`.pt` / `.pth`) or ONNX (`.onnx`) model file into this directory (e.g. `models/voxguard_model.pt`).

2. **Create Model Manifest**:
   Copy `manifest.example.json` to `models/manifest.json` and update the metadata:
   - Compute SHA-256 checksum:
     - PowerShell: `Get-FileHash -Algorithm SHA256 models/voxguard_model.pt`
     - Linux/macOS: `sha256sum models/voxguard_model.pt`
   - Ensure `checkpoint_path` points to your model file.
   - Verify `expected_sample_rate` is `16000` and `class_names` contains `["human", "synthetic"]`.

3. **Configure Environment Variables**:
   In `backend/.env` (or environment settings):
   ```env
   MODEL_ENABLED=True
   MODEL_PATH=models/voxguard_model.pt
   ```

4. **Automatic Offline Loading**:
   Upon server startup or test runner execution, VoxGuard's `ModelLoader` and `ModelValidator` will passively verify manifest integrity, file format, SHA-256 checksum, and tensor shape compatibility before setting `status = "ready"`.

If no model checkpoint exists, VoxGuard operates safely in **`ANALYSIS_ONLY`** mode with `probabilities = {"human": null, "synthetic": null}`.
