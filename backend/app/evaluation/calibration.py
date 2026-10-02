"""Probability Calibration Specification & Interface for VoxGuard.

Defines probability calibration descriptors and interfaces (temperature scaling,
Platt scaling, isotonic regression) while reporting status="not_calibrated" by default
until real evaluated calibration artifacts exist.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple


class CalibrationMethod(str, Enum):
    """Supported probability calibration algorithms."""
    NONE = "none"
    TEMPERATURE_SCALING = "temperature_scaling"
    PLATT_SCALING = "platt_scaling"
    ISOTONIC_REGRESSION = "isotonic_regression"


@dataclass
class CalibrationConfig:
    """Descriptor configuration for probability calibration."""
    method: CalibrationMethod = CalibrationMethod.NONE
    status: str = "not_calibrated"                  # "calibrated", "not_calibrated"
    parameters: Dict[str, Any] = field(default_factory=dict)
    notes: str = "No probability calibration artifact configured."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "status": self.status,
            "parameters": self.parameters,
            "notes": self.notes
        }


class ProbabilityCalibrator:
    """Service applying probability calibration scaling to raw model logits or probabilities."""

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()

    def calibrate(self, p_human: float, p_synthetic: float) -> Tuple[float, float]:
        """Applies calibration algorithm if calibrated status is active.

        Defaults to returning raw normalized probabilities when calibration_status is "not_calibrated".
        """
        if self.config.status != "calibrated" or self.config.method == CalibrationMethod.NONE:
            # Return raw probabilities un-calibrated
            total = p_human + p_synthetic
            if total > 0:
                return round(p_human / total, 4), round(p_synthetic / total, 4)
            return 0.5, 0.5

        if self.config.method == CalibrationMethod.TEMPERATURE_SCALING:
            temp = float(self.config.parameters.get("temperature", 1.0))
            if temp <= 0:
                temp = 1.0
            # Temperature scaling on log-probabilities
            import numpy as np
            logits = np.log(np.clip([p_human, p_synthetic], 1e-7, 1.0)) / temp
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            return round(float(probs[0]), 4), round(float(probs[1]), 4)

        total = p_human + p_synthetic
        return round(p_human / total, 4), round(p_synthetic / total, 4)
