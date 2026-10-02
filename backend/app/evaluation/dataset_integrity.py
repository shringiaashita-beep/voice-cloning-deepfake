"""Dataset Integrity & Cryptographic Checksum Engine for VoxGuard.

Calculates SHA-256 digests for generated feature artifacts (.npz) and maintains
features_manifest.json metadata without exposing absolute filesystem paths.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Any, Dict, List, Optional


class DatasetIntegrity:
    """Computes artifact checksums and produces features_manifest.json metadata."""

    @staticmethod
    def compute_file_sha256(file_path: str) -> str:
        """Computes SHA-256 hex digest of a local file.

        Args:
            file_path: Path to local file.

        Returns:
            SHA-256 hex string (lower-case 64 chars).
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def build_features_manifest_json(
        artifact_hashes: Dict[str, str],
        dataset_version: str = "1.0.0",
        feature_pipeline_version: str = "voxguard-features-0.1.0",
        generation_timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """Constructs features_manifest.json dictionary representation.

        Args:
            artifact_hashes: Dict mapping relative artifact path (e.g. 'train/sample.npz') to SHA-256 hex string.
            dataset_version: Version string for dataset.
            feature_pipeline_version: Version identifier for feature pipeline.
            generation_timestamp: ISO 8601 timestamp string (defaults to current UTC time).

        Returns:
            Dict containing manifest metadata.
        """
        if generation_timestamp is None:
            generation_timestamp = datetime.now(timezone.utc).isoformat()

        # Ensure all keys in artifact_hashes are strictly relative paths
        sanitized_hashes: Dict[str, str] = {}
        for rel_path, digest in artifact_hashes.items():
            norm_rel = rel_path.strip().replace("\\", "/")
            if norm_rel.startswith("/") or ":" in norm_rel:
                norm_rel = os.path.basename(norm_rel)
            sanitized_hashes[norm_rel] = digest

        return {
            "dataset_version": dataset_version,
            "feature_pipeline_version": feature_pipeline_version,
            "generation_timestamp": generation_timestamp,
            "sample_count": len(sanitized_hashes),
            "artifact_hashes": sanitized_hashes
        }
