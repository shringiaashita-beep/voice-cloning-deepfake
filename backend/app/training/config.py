"""Training Configuration Module for VoxGuard.

Defines strongly typed training batch preparation settings, windowing parameters,
and deterministic DataLoader options.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class TrainingConfig:
    """Strongly typed training data loader and batch preparation configuration."""
    TARGET_FRAMES: int = 256
    FEATURES: List[str] = field(
        default_factory=lambda: ["log_mel", "mfcc", "spectral_centroid", "spectral_rolloff"]
    )
    WINDOW_STRIDE: int = 128
    PADDING_MODE: str = "zero"
    TRUNCATION_MODE: str = "deterministic"
    RANDOM_SEED: int = 42
    BATCH_SIZE: int = 16
    SHUFFLE_TRAIN: bool = True
    SHUFFLE_VALIDATION: bool = False
    SHUFFLE_TEST: bool = False
    SHUFFLE_OOD: bool = False
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = False
    DROP_LAST: bool = False

    @property
    def WINDOW_SIZE(self) -> int:
        """Alias for TARGET_FRAMES for compatibility."""
        return self.TARGET_FRAMES

    def to_dict(self) -> Dict[str, Any]:
        """Returns dictionary representation of training configuration."""
        return asdict(self)
